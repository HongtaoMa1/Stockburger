# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""段切分、清洗与日历块账本。

规格：验收标准 v007（v007.1）的 A1（三段）、A3（清洗）、A2（测试段考一次）、B5（三值）、
B9 / B16（V 段账本：一段 V 可以给多个任务用，记复用、只加注）、B19（V 段默认紧挨测试段，用过也不挪）。
口径（A6′，U-6）：「过闸 / 未过闸」只作参考、不拦人；只有协议被破坏才出「不可认证」。
本模块管三件事：哪天归哪段；样本留不留（清洗）；哪段日历当过几个任务的 V、测试段被哪些需求考过。
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import gate as G

# ---------- 口径常量：输出值用中文，标识符用英文 ----------
PASSED, NOT_PASSED, UNCERTIFIABLE = "过闸", "未过闸", "不可认证"   # 闸门三值（B5）
PROTOCOL_OK = "协议完好"                                          # 本层没查出协议破坏
GATE_RESULTS = (PASSED, NOT_PASSED, UNCERTIFIABLE)

SEG_S, SEG_V, SEG_TEST = "S", "V", "测试"
SEGMENTS = (SEG_S, SEG_V, SEG_TEST)
GAP, OUTSIDE = "空档", "段外"

UNUSED, USED, EXAMINED = "未用", "已用", "已考"     # 日历块只有这三种状态（B16）
BY_HUMAN, BY_AI = "人", "AI 代确认"                 # 送考候选由谁确认（A2）
RETEST_NOTE = "在几个成绩里挑 = 在测试数据上挑"       # A2：同一段从第 2 次考起加注，不设闸
REUSE_NOTE, reuse_note = G.REUSE_NOTE, G.reuse_note  # B16（v007.1）：一段 V 从第 2 个任务起加注，不判不可认证
NO_CLEAN_V = "无干净 V 段"                          # 旧 B19（locate_v 用；v007.1 起默认流程不调用）

# 账本每行的 event
EV_REGISTER, EV_TOUCH, EV_ABORT, EV_FINISH, EV_EXAM = "登记V", "碰V", "中止V", "结束V", "开考"


def _pos_int(name: str, value) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} 必须是正整数，收到 {value!r}")
    return int(value)


# ======================================================================
# 落盘：只追加的 JSONL（试验账本和日历块账本共用）
# ======================================================================

def _jsonable(o):
    """json.dumps 的 default：numpy 数、日期、路径、集合转成 JSON 认得的样子。"""
    if isinstance(o, (pd.Timestamp, _dt.date)):
        t = pd.Timestamp(o)
        return str(t.date()) if t == t.normalize() else t.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "item"):                      # numpy 标量
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    return str(o)                               # Path、pd.Period 等


def normalize_row(row: dict) -> dict:
    """一行过一遍 JSON 往返：写进文件的和留在内存里的完全一样。"""
    return json.loads(json.dumps(row, ensure_ascii=False, default=_jsonable))


def jsonl_append(path, row: dict) -> None:
    """往 JSONL 文件末尾追加一行并落盘（flush + fsync），机器断电或蓝屏时已写的行不丢。
    文件末尾若是上次断电写了一半的残行，先补一个换行，新行不跟它粘在一起。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, default=_jsonable).encode("utf-8") + b"\n"
    with open(p, "a+b") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                line = b"\n" + line
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def jsonl_rows(path) -> list[dict]:
    """读回 JSONL 文件的全部行；文件不存在 = 空账。读不懂的行（断电时写了一半的残行）跳过。"""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def _now() -> str:
    return pd.Timestamp.now().isoformat(timespec="seconds")


# ======================================================================
# 一、切分（A1）与清洗（A3）
# ======================================================================

@dataclass(frozen=True)
class Segment:
    """一段：段名、首个与末个交易日、交易日数。"""
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_days: int


@dataclass(frozen=True, eq=False)
class SegmentSplit:
    """切分结果（A1：切分日期只能人定或取默认，顶层不能设也不能改）。

    dates 是运行时数据的全部交易日（已排序去重）；s 在历史放不下三段时为 None；
    warnings 里只放「闸门无力认证」一类提示（B5：只提示，任务照跑）。
    """
    dates: pd.DatetimeIndex
    s: Segment | None
    v: Segment
    test: Segment
    d_pur: int
    label_horizon: int
    warnings: tuple[str, ...] = ()

    @property
    def t_1(self) -> pd.Timestamp:
        """测试段首日 t_1。"""
        return self.test.start

    @property
    def t_2(self) -> pd.Timestamp:
        """测试段末日 t_2 = 运行时数据末日。"""
        return self.test.end

    def ranges(self) -> dict:
        """每段的日期范围 {段名: (首日, 末日)}；S 段放不下时为 None。"""
        return {
            SEG_S: None if self.s is None else (self.s.start, self.s.end),
            SEG_V: (self.v.start, self.v.end),
            SEG_TEST: (self.test.start, self.test.end),
        }

    def assign(self, dates) -> pd.Series:
        """按日期把样本归到 S / V / 测试段；段间空档标「空档」，数据范围外标「段外」。
        dates 可以有重复（同一天多只股票），返回的 Series 与输入逐个对齐。"""
        d = pd.DatetimeIndex(pd.to_datetime(dates))
        out = np.full(len(d), OUTSIDE, dtype=object)
        out[(d >= self.dates[0]) & (d <= self.dates[-1])] = GAP
        for seg in (self.s, self.v, self.test):
            if seg is not None:
                out[(d >= seg.start) & (d <= seg.end)] = seg.name
        return pd.Series(out, index=d, name="segment")

    def label_end(self, sample_dates, horizon: int | None = None) -> pd.DatetimeIndex:
        """样本日 t 的标签窗末端 = t 之后第 H 个交易日（Qlib 默认标签 H = 2：t+1 收盘到 t+2 收盘）。
        超出数据末日的记 NaT（标签还没实现，不能用）。"""
        h = self.label_horizon if horizon is None else _pos_int("horizon", horizon)
        d = pd.DatetimeIndex(pd.to_datetime(sample_dates))
        pos = self.dates.searchsorted(d, side="right") - 1 + h
        ok = (pos >= 0) & (pos < len(self.dates))
        out = np.full(len(d), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        out[ok] = self.dates.values[pos[ok]]
        return pd.DatetimeIndex(out)

    def purge_boundary(self, segment: str) -> pd.Timestamp:
        """A3 的清洗界 = 下一段首日：S 段样本的标签窗末端须早于 V 段首日，V 段样本的须早于测试段首日。"""
        if segment == SEG_S:
            if self.s is None:
                raise ValueError("S 段放不下（历史太短），没有 S 段清洗界")
            return self.v.start
        if segment == SEG_V:
            return self.test.start
        raise ValueError(f"只有 S 段和 V 段要清洗，收到 {segment!r}")

    def keep_before(self, sample_dates, boundary, horizon: int | None = None) -> np.ndarray:
        """A3 的通用一步：标签窗末端早于 boundary 的样本留下（标签还没实现的 NaT 不留）。
        S 段内部切验证折（purged CV、Qlib 给 LightGBM 早停用的 valid 段）时，训练样本按验证折首日照此清洗。"""
        ends = self.label_end(sample_dates, horizon)
        return ~np.asarray(ends.isna()) & np.asarray(ends < pd.Timestamp(boundary))

    def purge_keep(self, segment: str, sample_dates, horizon: int | None = None) -> np.ndarray:
        """A3：某段的样本留不留——样本日落在该段内、且标签窗末端早于下一段首日才留。
        照默认（段间空档 d_pur = 标签期）段内样本一个都不删。"""
        seg = {SEG_S: self.s, SEG_V: self.v}.get(segment)
        if seg is None:
            raise ValueError(f"只能给 S 段或 V 段算清洗（且该段要存在），收到 {segment!r}")
        d = pd.DatetimeIndex(pd.to_datetime(sample_dates))
        inside = np.asarray((d >= seg.start) & (d <= seg.end))
        return inside & self.keep_before(d, self.purge_boundary(segment), horizon)

    def purge_check(self, s_label_ends=None, v_label_ends=None) -> dict:
        """B5 的「清洗检查」：S 段训练样本的标签窗末端都早于 V 段首日、V 段样本的都早于测试段首日。
        不满足（标签终点缺失 NaT 也算）即「不可认证」——等于训练偷看了下一段。
        返回 {"verdict": 协议完好 / 不可认证, "reasons": [...]}。"""
        reasons = []
        for seg, ends in ((SEG_S, s_label_ends), (SEG_V, v_label_ends)):
            if ends is None:
                continue
            e = pd.DatetimeIndex(pd.to_datetime(ends))
            if len(e) == 0:
                continue
            b = self.purge_boundary(seg)
            bad = np.asarray(e.isna()) | np.asarray(e >= b)
            if bad.any():
                reasons.append(f"{seg} 段有 {int(bad.sum())} 个样本的标签窗末端不早于下一段首日 {b.date()}（A3）")
        return {"verdict": UNCERTIFIABLE if reasons else PROTOCOL_OK, "reasons": reasons}

    def fingerprint(self) -> str:
        """切分的哈希（B3：碰 V 前登记时存下，事后比对就知道有没有人改过切分日期）。
        算法不能改：改了，已登记的指纹就对不上了。"""
        payload = {
            "ranges": {
                k: None if v is None else [str(v[0].date()), str(v[1].date())]
                for k, v in self.ranges().items()
            },
            "d_pur": self.d_pur,
            "label_horizon": self.label_horizon,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def split_segments(dates, t_val: int = 252, t_test: int = 252, label_horizon: int = 2,
                   d_pur: int | None = None, min_s_days: int = 1) -> SegmentSplit:
    """A1：把运行时数据的交易日切成 S 段 < V 段 < 测试段。

    - 测试段 = 最近 t_test 个交易日，t_2 = 数据末日；
    - 往前空 d_pur 个交易日，再往前 t_val 个交易日是 V 段；再空 d_pur 个交易日，之前全归 S 段；
    - d_pur ≥ 标签期 label_horizon；不给就取 d_pur = 标签期（默认不另加 embargo）。
    t_val = t_test = 252 是 [GUESS] 默认，由人定，顶层不能定也不能改。
    历史连 V 段加测试段都放不下 → ValueError；放得下但 S 段不足 min_s_days 个交易日 →
    挂「闸门无力认证」警告，只提示，任务照跑（B5）。
    """
    t_val = _pos_int("t_val", t_val)
    t_test = _pos_int("t_test", t_test)
    label_horizon = _pos_int("label_horizon", label_horizon)
    min_s_days = _pos_int("min_s_days", min_s_days)
    d_pur = label_horizon if d_pur is None else int(d_pur)
    if d_pur < label_horizon:
        raise ValueError(f"段间空档 d_pur = {d_pur} 小于标签期 {label_horizon}（A1：d_pur ≥ 标签期）")

    idx = pd.DatetimeIndex(pd.to_datetime(dates)).dropna().unique().sort_values()
    n = len(idx)
    t0 = n - t_test                  # 测试段首日下标（t_1）
    v_end = t0 - d_pur - 1
    v_start = v_end - t_val + 1
    if t0 < 0 or v_start < 0:
        need = t_test + d_pur + t_val
        raise ValueError(f"历史放不下 V 段和测试段：至少要 {need} 个交易日，只有 {n} 个")
    s_end = v_start - d_pur - 1

    warnings = []
    s = Segment(SEG_S, idx[0], idx[s_end], s_end + 1) if s_end >= 0 else None
    n_s = 0 if s is None else s.n_days
    if n_s < min_s_days:
        warnings.append(
            f"闸门无力认证：历史放不下三段（S 段只有 {n_s} 个交易日，要求 ≥ {min_s_days}）——只提示，任务照跑（B5）"
        )
    v = Segment(SEG_V, idx[v_start], idx[v_end], t_val)
    test = Segment(SEG_TEST, idx[t0], idx[n - 1], t_test)
    return SegmentSplit(idx, s, v, test, d_pur, label_horizon, tuple(warnings))


# ======================================================================
# 二、日历块账本（B9 / B16 / B19 / A2）
# ======================================================================

def _blocks_text(blocks) -> str:
    """一串日历块的短写：不多于 3 个全列，多了写「首…末（n 块）」。"""
    b = [str(x) for x in blocks]
    return "、".join(b) if len(b) <= 3 else f"{b[0]}…{b[-1]}（{len(b)} 块）"


class BlockLedger:
    """V 段与测试段共用的日历块账本（B9 / B16 / B19 / A2；v007.1）。

    键是「市场 × 日历块」（块长 block_freq 开跑前登记，默认月 "M"；区间只盖住半个块也算含该块）。
    块只有三种状态：未用 / 已用（给过任务当 V）/ 已考（作测试段被考过）。
    一段 V 可以给多个任务用（v007.1）：用过的块照登记，账本记复用信息——每块给过几个任务、
    这次登记是这段 V 的第几个任务 n（按块取最大，和考试的「第几场」一个算法）。n ≥ 2 时加注
    REUSE_NOTE（「这段 V 给过 n 个任务：闸门结论更只能当参考」），只是加注，不判不可认证。
    - register_v：登记 V，块记「已用」、这个任务计一次；回执带复用信息；
    - touch_v：首次读 V 段数据之前报到；
    - abort_v：碰 V 前中止 → 这个任务退出计数（块没别的任务用就退回「未用」）；碰 V 后中止、或 finish_v 正常结束 → 照计；
    - check_v：闸门动用 V 之前查账，带出复用信息（n、加注）；V 没在账本登记、或碰 V 前中止过又拿来用 →「不可认证」，
      归 B5 的「登记被改」（等于没登记就碰 V，账本数不出这段 V 给过几个任务）；
    - start_exam：考试脚本读测试段之前报到，准考就记「已考」（A2：每条需求考一次）；
    - locate_v：往过去找最近一段块都「未用」的 V 窗。v007.1 起默认流程不调用（V 默认紧挨测试段、用过也不挪，B19），
      留给按旧规矩挪过 V 的登记复核用。
    账本写进 JSONL 文件 path：只追加、跨任务共用，每行带任务 ID；每次查账都从文件重读，别的任务（别的进程）
    刚写的行也看得到；几个任务登记同一段 V 时，按文件里的先后排第几个。path=None 时只记在内存里。
    """

    def __init__(self, path=None, block_freq: str = "M"):
        try:
            pd.Period("2000-01-01", freq=block_freq)
        except (TypeError, ValueError) as err:
            raise ValueError(f"认不出日历块长 block_freq = {block_freq!r}（用 pandas 周期写法，如 'M'、'Q'、'W'）") from err
        self.path = None if path is None else Path(path)
        self.block_freq = block_freq
        self._mem: list[dict] = []
        self._load()

    # ---------- 重放 ----------
    def _load(self) -> None:
        """从头重放全部行，重建账面（文件模式每次查账前都重读）。"""
        self._tasks: dict[str, dict] = {}
        self._holders: dict[tuple, list] = {}       # (市场, 块) → 拿它当 V 的任务，按登记先后（碰 V 前中止的已退出）
        self._exams: list[dict] = []
        self._exam_at: dict[tuple, list] = {}       # (市场, 块) → 考过它的 _exams 下标
        for row in (self._mem if self.path is None else jsonl_rows(self.path)):
            self._apply(row)

    def _apply(self, row: dict) -> None:
        ev, tid = row.get("event"), str(row.get("task_id"))
        if ev == EV_REGISTER:
            if tid in self._tasks:
                return                                      # 同一任务 ID 登记两次：先写的算数
            self._tasks[tid] = {"task_id": tid, "market": row["market"], "start": row["start"], "end": row["end"],
                                "blocks": list(row["blocks"]), "touched": False, "phase": "登记",
                                "result": None, "refunded": False, "seq": len(self._tasks)}
            for b in row["blocks"]:                         # 用过的块照登记（B16，v007.1）：排在已有任务后面
                self._holders.setdefault((row["market"], b), []).append(tid)
        elif ev in (EV_TOUCH, EV_ABORT, EV_FINISH) and tid in self._tasks:
            t = self._tasks[tid]
            if ev == EV_TOUCH:
                t["touched"] = True
            elif ev == EV_FINISH:
                t["phase"], t["result"] = "结束", row.get("result")
            else:
                t["phase"] = "中止"
                if not t["touched"]:                        # 碰 V 前中止：退出计数（块没别的任务用就退回未用）
                    t["refunded"] = True
                    for b in t["blocks"]:
                        holders = self._holders.get((t["market"], b), [])
                        if tid in holders:
                            holders.remove(tid)
        elif ev == EV_EXAM:
            i = len(self._exams)
            self._exams.append(dict(row))
            for b in row["blocks"]:
                self._exam_at.setdefault((row["market"], b), []).append(i)

    def _write(self, row: dict) -> dict:
        row = normalize_row({**row, "time": _now()})
        if self.path is None:
            self._mem.append(row)
        else:
            jsonl_append(self.path, row)
        self._apply(row)
        return row

    def _refresh(self) -> None:
        if self.path is not None:
            self._load()

    # ---------- 小工具 ----------
    @staticmethod
    def _span(start, end) -> tuple:
        s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        if e < s:
            raise ValueError(f"区间末日 {e.date()} 早于首日 {s.date()}")
        return s, e

    def blocks(self, start, end) -> list[str]:
        """[start, end] 盖到的日历块（盖住一部分也算），如 ["2024-10", "2024-11"]。"""
        s, e = self._span(start, end)
        f = self.block_freq
        return [str(p) for p in pd.period_range(pd.Period(s, freq=f), pd.Period(e, freq=f), freq=f)]

    def _state(self, market: str, block: str) -> str:
        if self._holders.get((market, block)):
            return USED
        return EXAMINED if self._exam_at.get((market, block)) else UNUSED

    def _window_state(self, market: str, blocks) -> str:
        """一串块里最重的状态（已用 > 已考 > 未用）。"""
        states = {self._state(market, b) for b in blocks}
        return next(st for st in (USED, EXAMINED, UNUSED) if st in states)

    def _reuse(self, t: dict) -> dict:
        """一个任务的复用信息（B16，v007.1；以账本重放为准）：
        n = 这段 V 的第几个任务（按块取最大：某块此前给过 k 个任务，这块上它就是第 k + 1 个）；
        prior_uses = 每块此前给过几个任务；reused_from = 此前拿其中块当过 V 的任务（按登记先后）；
        note = n ≥ 2 时的加注，否则空串。碰 V 前中止、已退出计数的任务没有 n（None）。"""
        if t["refunded"]:
            return {"n": None, "prior_uses": {}, "reused_from": [], "note": ""}
        prior, before = {}, set()
        for b in t["blocks"]:
            holders = self._holders.get((t["market"], b), [])
            k = holders.index(t["task_id"]) if t["task_id"] in holders else len(holders)
            prior[b] = k
            before.update(holders[:k])
        n = max(prior.values(), default=0) + 1
        return {"n": n, "prior_uses": prior, "reused_from": sorted(before, key=lambda u: self._tasks[u]["seq"]),
                "note": reuse_note(n)}

    def _open_task(self, task_id) -> dict:
        t = self._tasks.get(str(task_id))
        if t is None:
            raise ValueError(f"任务 {task_id} 没在账本里登记 V")
        if t["phase"] != "登记":
            raise ValueError(f"任务 {task_id} 已经{t['phase']}")
        return t

    # ---------- V 段 ----------
    def register_v(self, market, start, end, task_id) -> dict:
        """登记 V（B16，v007.1：一段 V 可以给多个任务用）：用过的块照登记，块记「已用」、这个任务计一次，不判不可认证、
        不拦登记。回执除登记信息外带复用信息：n（这次是这段 V 的第几个任务，按块取最大）、prior_uses（每块此前给过
        几个任务）、reused_from（此前拿其中块当过 V 的任务）、note（n ≥ 2 时的加注「这段 V 给过 n 个任务：闸门结论
        更只能当参考」）。窗口含被考过的块也照登记（不判不可认证），回执里列出来。几个进程同时登记同一段时，各自回执
        里的 n 只算到当时看到的账；以文件重放为准（闸门前用 check_v / touch_v 再查）。
        断了重跑时同一任务、同一窗口再登记一次，照原登记接着用，不另写行。"""
        self._refresh()
        tid, market = str(task_id), str(market)
        s, e = self._span(start, end)
        blocks = self.blocks(s, e)
        examined = [b for b in blocks if self._exam_at.get((market, b))]
        out = {"verdict": PROTOCOL_OK, "registered": False, "task_id": tid, "market": market, "start": s, "end": e,
               "blocks": blocks, "examined_blocks": examined, "reasons": [], "warnings": [], "resumed": False}
        if tid in self._tasks:
            t = self._tasks[tid]
            if (t["market"], t["start"], t["end"]) == (market, str(s.date()), str(e.date())) and t["phase"] == "登记":
                return {**out, **self._reuse(t), "registered": True, "resumed": True,
                        "warnings": ["这个任务已经登记过这段 V（断了重跑）：照原登记接着用"]}
            raise ValueError(f"task_id {tid} 已经在账本里登记过")
        self._write({"event": EV_REGISTER, "task_id": tid, "market": market, "start": s, "end": e, "blocks": blocks})
        info = self._reuse(self._tasks[tid])
        if info["n"] >= 2:
            used = [b for b in blocks if info["prior_uses"][b]]
            out["warnings"].append(f"V 里有用过的块 {_blocks_text(used)}（给过任务 {'、'.join(info['reused_from'])}）："
                                   f"照登记，这次是这段 V 的第 {info['n']} 个任务——{info['note']}（B16，不判不可认证）")
        if examined:
            out["warnings"].append(f"V 里有被考过的块 {_blocks_text(examined)}：照用，不判不可认证（B19）")
        return {**out, **info, "registered": True}

    def touch_v(self, task_id) -> dict:
        """首次读 V 段数据之前报到：此后再中止，这个任务也照计、不退出。返回 check_v 的结果（带复用信息）。"""
        self._refresh()
        t = self._open_task(task_id)
        if not t["touched"]:
            self._write({"event": EV_TOUCH, "task_id": t["task_id"]})
        return self.check_v(task_id)

    def check_v(self, task_id) -> dict:
        """闸门动用 V 段之前查账（B5 / B16）。返回 {"verdict", "task_id", "reasons", "n", "prior_uses", "reused_from",
        "note"}：复用信息以账本重放为准，n ≥ 2 带加注，只作参考、不判不可认证（v007.1）。
        V 没在账本登记、或碰 V 前中止过（已退出计数）又拿来跑闸门 →「不可认证」，归 B5 的「登记被改」：
        等于没登记就碰 V，账本数不出这段 V 给过几个任务。"""
        self._refresh()
        tid = str(task_id)
        t = self._tasks.get(tid)
        if t is None:
            info = {"n": None, "prior_uses": {}, "reused_from": [], "note": ""}
            reasons = [f"登记被改（B5）：任务 {tid} 的 V 没在日历块账本登记（B9 / B16），等于没登记就碰 V"]
        else:
            info = self._reuse(t)
            reasons = ([f"登记被改（B5）：任务 {tid} 碰 V 前中止过、V 已退回（B16），又拿来跑闸门，等于没登记就碰 V"]
                       if t["refunded"] else [])
        return {"verdict": UNCERTIFIABLE if reasons else PROTOCOL_OK, "task_id": tid, "reasons": reasons, **info}

    def finish_v(self, task_id, result: str) -> dict:
        """任务以闸门结果结束（B16）：照计，块保持「已用」。只收三值；没碰过 V 就结束请用 abort_v。"""
        if result not in GATE_RESULTS:
            raise ValueError(f"闸门结果只能是 {GATE_RESULTS}，收到 {result!r}")
        self._refresh()
        t = self._open_task(task_id)
        if not t["touched"]:
            raise ValueError(f"任务 {task_id} 还没碰过 V：没碰 V 就结束请用 abort_v（退出计数）")
        self._write({"event": EV_FINISH, "task_id": t["task_id"], "result": result})
        info = self._reuse(t)
        return {"task_id": t["task_id"], "result": result, "blocks": list(t["blocks"]), "state": USED,
                "n": info["n"], "note": info["note"]}

    def abort_v(self, task_id) -> dict:
        """任务中止（B16）：碰 V 前中止 → 这个任务退出计数（块没别的任务用就退回「未用」）；碰 V 后中止 → 照计、块保持「已用」。
        state 是中止后这段 V 的账面（各块里最重的状态）。"""
        self._refresh()
        t = self._open_task(task_id)
        self._write({"event": EV_ABORT, "task_id": t["task_id"]})
        return {"task_id": t["task_id"], "refunded": t["refunded"], "blocks": list(t["blocks"]),
                "state": self._window_state(t["market"], t["blocks"])}

    # ---------- 测试段（A2）----------
    def start_exam(self, market, start, end, *, demand_id, task_id, cand_id, confirmed_by: str = BY_HUMAN) -> dict:
        """考试脚本读测试段之前报到（A2）：只收一个候选；确认人是「人」或「AI 代确认」（没人时按碰 V 前登记的
        挑选规则代为确认，日报标「AI 代确认」）。同一需求在相交的区间（按日历块算）考过 → 不准考、「不可认证」；
        换一条需求可以再考同一段，从第 2 次起加注 RETEST_NOTE，不设闸。准考就写一行，块记「已考」。
        断了重跑同一场（同一任务、需求、候选、区间）不算再考：照原来那场接着考，不另写行。
        返回 {"verdict", "allowed", "n": 此前考过几次（按块取最大）, "k": 本场是第几场, "note", "reasons"}。"""
        if isinstance(cand_id, (list, tuple, set, frozenset, dict)):
            raise ValueError("考试只收一个候选（A2）")
        if confirmed_by not in (BY_HUMAN, BY_AI):
            raise ValueError(f"确认人只能是「{BY_HUMAN}」或「{BY_AI}」，收到 {confirmed_by!r}")
        self._refresh()
        market = str(market)
        s, e = self._span(start, end)
        blocks = self.blocks(s, e)
        same = (str(task_id), str(demand_id), str(cand_id), market, str(s.date()), str(e.date()))
        for x in self._exams:
            if (x["task_id"], x["demand_id"], x["cand_id"], x["market"], x["start"], x["end"]) == same:
                return {"verdict": PROTOCOL_OK, "allowed": True, "n": x["k"] - 1, "k": x["k"], "note": x["note"],
                        "confirmed_by": x["confirmed_by"], "reasons": [], "resumed": True}
        n = max(len(self._exam_at.get((market, b), [])) for b in blocks)
        before = {i for b in blocks for i in self._exam_at.get((market, b), [])}
        if any(self._exams[i]["demand_id"] == str(demand_id) for i in before):
            return {"verdict": UNCERTIFIABLE, "allowed": False, "n": n, "k": None, "note": "",
                    "confirmed_by": confirmed_by, "resumed": False,
                    "reasons": [f"需求 {demand_id} 已在相交的区间考过一次：同一需求再考 = 不可认证（A2）"]}
        k = n + 1
        note = RETEST_NOTE if k >= 2 else ""
        self._write({"event": EV_EXAM, "task_id": str(task_id), "demand_id": str(demand_id), "cand_id": str(cand_id),
                     "market": market, "start": s, "end": e, "blocks": blocks, "confirmed_by": confirmed_by,
                     "k": k, "note": note})
        return {"verdict": PROTOCOL_OK, "allowed": True, "n": n, "k": k, "note": note,
                "confirmed_by": confirmed_by, "reasons": [], "resumed": False}

    # ---------- 查询 ----------
    def block_status(self, market, block) -> dict:
        """一个日历块的账面：状态、拿它当 V 的任务（列表，按登记先后）、给过几个任务 n_uses、被考过几次、考过它的需求。"""
        self._refresh()
        key = (str(market), str(pd.Period(block, freq=self.block_freq)))
        exams = [self._exams[i] for i in self._exam_at.get(key, [])]
        holders = list(self._holders.get(key, []))
        return {"market": key[0], "block": key[1], "state": self._state(*key), "used_by": holders,
                "n_uses": len(holders), "n_exams": len(exams), "demands": [x["demand_id"] for x in exams]}

    def status(self, market, start, end) -> dict:
        """一段日期的总账：state 取各块里最重的（已用 > 已考 > 未用）；used_by = 拿其中块当过 V 的任务（按登记先后）；
        复用信息（v007.1）：block_uses = 每块给过几个任务，n_uses 取其中最大；n_exams 取各块被考次数的最大值。"""
        self._refresh()
        market = str(market)
        s, e = self._span(start, end)
        blocks = self.blocks(s, e)
        states = {b: self._state(market, b) for b in blocks}
        uses = {b: len(self._holders.get((market, b), [])) for b in blocks}
        users = {u for b in blocks for u in self._holders.get((market, b), [])}
        n = max(len(self._exam_at.get((market, b), [])) for b in blocks)
        return {"market": market, "start": s, "end": e, "blocks": blocks, "state": self._window_state(market, blocks),
                "block_states": states, "used_by": sorted(users, key=lambda u: self._tasks[u]["seq"]),
                "block_uses": uses, "n_uses": max(uses.values()), "n_exams": n}

    def rows(self) -> list[dict]:
        """账本全部行（副本），按写入先后。"""
        self._refresh()
        return [dict(r) for r in (self._mem if self.path is None else jsonl_rows(self.path))]

    # ---------- 旧 B19 的 V 段定位（v007.1 起默认流程不调用）----------
    def locate_v(self, market, dates, *, t_val: int, test_start, d_pur: int, min_s_days: int = 1,
                 n_tests: int = 1, alpha: float = 0.05, beta: float = 0.2) -> dict:
        """旧 B19 的 V 段定位：从紧挨测试段的那段起往前找最近一段长 t_val 个交易日、连续、块都「未用」（没用过也没被
        考过）的区间，并保证 S < V < 测试段：V 末日与测试段首日之间、V 首日之前各空 d_pur 个交易日，再往前至少
        min_s_days 天 S 段。找不到就印「无干净 V 段」，列三个选项——缩短 V（回显 SR_mde）/ 照用被考过的块 / 换市场；
        例程不替人选。v007.1 起 V 默认紧挨测试段、用过也不往前挪（块照登记、记复用，B16 / B19），默认流程不再调用本例程；
        留着给按旧规矩挪过 V 的登记复核用。dates = 该市场运行时数据的全部交易日。"""
        self._refresh()
        market = str(market)
        idx = pd.DatetimeIndex(pd.to_datetime(dates)).dropna().unique().sort_values()
        t_val, min_s_days, d_pur = _pos_int("t_val", t_val), _pos_int("min_s_days", min_s_days), int(d_pur)
        if d_pur < 0:
            raise ValueError(f"d_pur 不能为负，收到 {d_pur}")
        lo = min_s_days + d_pur                                            # V 首日下标的下限
        hi = int(idx.searchsorted(pd.Timestamp(test_start))) - d_pur - 1   # V 末日下标的上限
        states = {}
        for i in range(lo, hi + 1):
            b = str(pd.Period(idx[i], freq=self.block_freq))
            if b not in states:
                states[b] = self._state(market, b)
        day_state = [states[str(pd.Period(idx[i], freq=self.block_freq))] for i in range(lo, hi + 1)]

        def latest_run(ok) -> tuple:
            """(最近一段长 ≥ t_val 的连续区间的末端, 最长连续段长, 它的末端)，下标相对 lo。"""
            run, latest, best, best_end = 0, None, 0, None
            for k, flag in enumerate(ok):
                run = run + 1 if flag else 0
                if run >= t_val:
                    latest = k
                if run > best:
                    best, best_end = run, k
            return latest, best, best_end

        def window(end_k: int) -> tuple:
            return idx[lo + end_k - t_val + 1], idx[lo + end_k]

        end_k, longest, longest_end = latest_run([st == UNUSED for st in day_state])
        base = {"market": market, "t_val": t_val}
        if end_k is not None:
            s, e = window(end_k)
            return {**base, "found": True, "start": s, "end": e, "blocks": self.blocks(s, e),
                    "message": f"最近一段干净的 V 窗：{s.date()} … {e.date()}"}
        if longest:
            a, b = idx[lo + longest_end - longest + 1], idx[lo + longest_end]
            shorten = (f"缩短 V：最长的干净连续段 {longest} 个交易日（{a.date()} … {b.date()}），"
                       f"SR_mde = {G.sr_mde(n_tests, longest, alpha, beta):.2f}（N = {n_tests}）")
        else:
            shorten = "缩短 V：能放 V 的范围里没有一个干净的交易日"
        relaxed = latest_run([st != USED for st in day_state])[0]
        if relaxed is not None:
            s, e = window(relaxed)
            reuse = f"照用被考过的块（不判不可认证）：{s.date()} … {e.date()}"
        else:
            reuse = "照用被考过的块（不判不可认证）：也凑不出一段"
        return {**base, "found": False, "message": NO_CLEAN_V, "options": [shorten, reuse, "换市场"],
                "longest_clean": longest}
