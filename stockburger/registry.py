# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""试验登记与试验账本。

规格：验收标准 v007 的 B3（碰 V 前登记）、B5（不可认证）、B7（试验账本）、
A2（测试段只有考试脚本读）。
口径（A6′，U-6）：本模块只判「协议完好 / 不可认证」；「过闸 / 未过闸」由闸门出，只作参考。

用法（默认 L = 0）：
    reg = TrialRegistry("task-001", path="cache/trials.jsonl")    # 跨任务共用的一本账
    reg.record("c1", status=DONE, proposal_text="...", scores={"S_DSR": 0.9}, s_returns=s_ret)
    reg.record("p9", status=REJECTED)                     # 碰数据前被校验拒掉 → 计 n_prop
    reg.register(n_fin=5, r_max=3, alpha_gate=0.05, ...)   # 碰 V 段前登记
    reg.enter_v("c1")                                     # 送检报到
    reg.append_info("c1", v_returns=v_ret)                 # 后来才有的信息，只填空字段
    reg.problems(); reg.summary()                          # 协议破坏清单；N_T、n_prop、送检数
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .segments import (PROTOCOL_OK, SEG_S, SEG_TEST, SEG_V, SEGMENTS, UNCERTIFIABLE, jsonl_append, jsonl_rows,
                       normalize_row)

# 候选状态（B7）
DONE = "完成"
ABORTED = "中止"
ABANDONED = "被放弃"
REJECTED = "校验拒绝"      # 碰数据前就被拒的提议：另计 n_prop，不进 N_T
STATUSES = (DONE, ABORTED, ABANDONED, REJECTED)

# 账本行的字段（B7）：id、任务 ID、轮次、父候选、提议哈希、碰了哪段数据、各段得分、状态、S 段与 V 段的逐日候选收益
ROW_FIELDS = ("cand_id", "task_id", "round", "parent_id", "proposal_hash", "touched", "scores", "status",
              "s_returns", "v_returns")

# 文件里每行的 event
EV_REGISTER, EV_TRIAL, EV_SEND, EV_INFO = "登记", "试验", "送检", "补记"

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _pos_int(name: str, value) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} 必须是正整数，收到 {value!r}")
    return int(value)


def _now() -> str:
    return pd.Timestamp.now().isoformat(timespec="seconds")


def _returns_json(x):
    """逐日候选收益怎么进账本：文件路径存成字符串；按日期索引的 Series 存成 {日期: 收益}（缺值不存）；
    没有日期的数组存成列表（缺值记 null）。"""
    if x is None:
        return None
    if isinstance(x, (str, os.PathLike)):
        return str(x)
    if isinstance(x, pd.Series) and isinstance(x.index, pd.DatetimeIndex):
        s = x.dropna()
        return {str(t.date()): float(v) for t, v in zip(s.index, s.values)}
    return [float(v) if np.isfinite(v) else None for v in np.asarray(x, dtype=float).ravel()]


def _returns_out(v):
    """读回：字符串是文件路径，照给；{日期: 收益} 还原成按日期索引的 pd.Series；列表还原成数组。"""
    if isinstance(v, dict):
        s = pd.Series(v, dtype=float)
        if all(_ISO_DATE.match(str(k)) for k in v):
            s.index = pd.to_datetime(s.index)
        return s
    if isinstance(v, list):
        return np.array([np.nan if x is None else x for x in v], dtype=float)
    return v


class TrialRegistry:
    """一个任务的登记（B3）与试验账本（B7），并按 B5 判协议破没破。

    - 试验账本只追加：每个候选一行（中止、被放弃的分支也记；碰数据前就被拒的提议也记一行，另计 n_prop），
      行不删不改；后来才有的信息（V 段得分、V 段收益）用 append_info 补记——只填空字段，在文件里是另起的一行。
    - 碰 V 前登记（B3）：register 写死送检数 n_fin、轮数上限 R_max 和 B3 其余各项（切分日期或指纹、d_pur、
      数据包版本、α_gate、β_pow、T_val、k_S、用哪个 t、L）；同样内容再登记无事，内容不同 = 登记被改。
    - 不可认证（B5）：碰 V 时还没登记、登记被改、送检数超出 n_fin、轮数超出 R_max、记录碰了测试段（A2）。
    path 给了就写进 JSONL 文件：跨任务共用一本账，每行带任务 ID，本实例只数自己任务的行；
    重开同一个文件会照文件重放，断了能接着记。path=None 只记在内存里。
    """

    def __init__(self, task_id, path=None):
        self.task_id = str(task_id)
        self.path = None if path is None else Path(path)
        self._registration: dict | None = None
        self._reg_hash: str | None = None
        self._rows: dict[str, dict] = {}          # cand_id → 合并后的一行（按首次记账先后）
        self._events: list[str] = []              # 重放时按先后判出的协议破坏
        if self.path is not None:
            for row in jsonl_rows(self.path):
                if str(row.get("task_id")) == self.task_id:
                    self._apply(row)

    # ---------- 落盘与重放 ----------
    def _apply(self, row: dict) -> None:
        ev, cid = row.get("event"), row.get("cand_id")
        if ev == EV_REGISTER:
            if self._registration is None:
                self._registration, self._reg_hash = row["registration"], row["hash"]
            elif row["hash"] != self._reg_hash:
                self._events.append(f"登记被改：原登记哈希 {self._reg_hash[:12]}…，新内容哈希 {row['hash'][:12]}…；"
                                    "仍按原登记算（B3 / B5）")
        elif ev == EV_TRIAL and cid not in self._rows:
            r = {k: row.get(k) for k in ROW_FIELDS}
            r["touched"] = list(row.get("touched") or [])
            r["scores"] = dict(row.get("scores") or {})
            self._rows[cid] = r
            if SEG_V in r["touched"]:
                self._note_v(cid)
        elif ev == EV_SEND and cid in self._rows:
            r = self._rows[cid]
            if SEG_V not in r["touched"]:
                r["touched"].append(SEG_V)
                self._note_v(cid)
        elif ev == EV_INFO and cid in self._rows:
            r = self._rows[cid]
            r["scores"].update(row.get("scores") or {})
            for k in ("s_returns", "v_returns"):
                if row.get(k) is not None:
                    r[k] = row[k]

    def _note_v(self, cid: str) -> None:
        if self._registration is None:
            self._events.append(f"登记被改（B5）：候选 {cid} 碰 V 段时还没登记（B3：碰 V 前登记；"
                                "没登记就碰 V，登记就能照 V 段结果事后定）")

    def _write(self, row: dict) -> None:
        row = normalize_row({**row, "task_id": self.task_id, "time": _now()})
        if self.path is not None:
            jsonl_append(self.path, row)
        self._apply(row)

    # ---------- 登记（B3）----------
    def register(self, n_fin: int, *, r_max: int | None = None, **extra) -> str:
        """碰 V 段前登记（B3）：n_fin = 送检数（L = 0 时也就是 N_T^tr），r_max = 轮数上限；B3 其余各项放进
        extra，一起进登记哈希。登记写死：同样内容再登记无事；内容不同 = 登记被改 → 不可认证，仍按原登记算。
        返回登记哈希。"""
        reg = normalize_row({"task_id": self.task_id, "n_fin": _pos_int("n_fin", n_fin),
                             "r_max": None if r_max is None else _pos_int("r_max", r_max), **extra})
        h = _sha256(json.dumps(reg, sort_keys=True, ensure_ascii=False))
        if h != self._reg_hash:
            self._write({"event": EV_REGISTER, "registration": reg, "hash": h})
        return self._reg_hash

    @property
    def registration(self) -> dict | None:
        """当前生效的登记（副本）；没登记为 None。"""
        return None if self._registration is None else dict(self._registration)

    @property
    def registration_hash(self) -> str | None:
        """登记哈希（碰 V 前写死，事后比对）。"""
        return self._reg_hash

    # ---------- 记账（B7）----------
    def record(self, cand_id, *, status: str, round_no: int = 1, parent_id=None, proposal_text=None,
               touched=None, scores=None, s_returns=None, v_returns=None) -> dict:
        """记一个候选（B7：只追加，一个候选一行）。

        proposal_text 只存哈希、不存原文；touched = 碰了哪段数据（不给时校验拒绝取空、其余取 ("S",)）；
        scores = 各段得分；s_returns / v_returns = S 段与 V 段的逐日候选收益（按日期索引的 Series，或文件路径）。
        status 取「完成 / 中止 / 被放弃 / 校验拒绝」；校验拒绝 = 碰数据前就被拒，touched 必须为空。
        touched 含 "V" 等于记账时就送检报到（同 enter_v）。返回这一行（副本）。
        """
        cid = str(cand_id)
        if cid in self._rows:
            raise ValueError(f"候选 {cid} 已有记录：账本只追加、一个候选一行；后补信息用 append_info")
        if status not in STATUSES:
            raise ValueError(f"status 只能是 {STATUSES}，收到 {status!r}")
        if touched is None:
            touched = () if status == REJECTED else (SEG_S,)
        touched = list(dict.fromkeys(touched))
        bad = [t for t in touched if t not in SEGMENTS]
        if bad:
            raise ValueError(f"碰了哪段数据只能是 {SEGMENTS}，收到 {bad}")
        if status == REJECTED and touched:
            raise ValueError("校验拒绝 = 碰数据前就被拒，touched 必须为空（这类提议计 n_prop）")
        self._write({
            "event": EV_TRIAL, "cand_id": cid, "round": _pos_int("round_no", round_no),
            "parent_id": None if parent_id is None else str(parent_id),
            "proposal_hash": None if proposal_text is None else _sha256(proposal_text),
            "touched": touched, "scores": dict(scores or {}), "status": status,
            "s_returns": _returns_json(s_returns), "v_returns": _returns_json(v_returns),
        })
        return self._copy(self._rows[cid])

    def enter_v(self, cand_id) -> str:
        """候选送进 V 段（闸门）之前报到：给它的「碰了哪段数据」添上 V，并按登记查上限。
        同一候选重复报到不重复计数。返回报到后整本账的结论（协议完好 / 不可认证）。"""
        row = self._rows.get(str(cand_id))
        if row is None:
            raise KeyError(f"候选 {cand_id} 不在账本里：先 record 再送检")
        if row["status"] == REJECTED:
            raise ValueError(f"候选 {cand_id} 是校验拒绝的提议，不能送检")
        if SEG_V not in row["touched"]:
            self._write({"event": EV_SEND, "cand_id": row["cand_id"]})
        return self.verdict

    def append_info(self, cand_id, *, scores=None, s_returns=None, v_returns=None) -> dict:
        """给已有的候选补信息（B7「只追加」）：只能填空字段——新的得分键、空着的收益；
        改已写的值 → ValueError。在文件里另起一行（event = 补记），原来那行不动。返回合并后的这一行（副本）。"""
        row = self._rows.get(str(cand_id))
        if row is None:
            raise KeyError(f"候选 {cand_id} 不在账本里")
        scores = dict(scores or {})
        clash = [k for k in scores if k in row["scores"]]
        if clash:
            raise ValueError(f"得分 {clash} 已写过，账本只追加、不改")
        info = {"event": EV_INFO, "cand_id": row["cand_id"]}
        if scores:
            info["scores"] = scores
        for name, val in (("s_returns", s_returns), ("v_returns", v_returns)):
            if val is not None:
                if row[name] is not None:
                    raise ValueError(f"{name} 已写过，账本只追加、不改")
                info[name] = _returns_json(val)
        self._write(info)
        return self._copy(row)

    # ---------- 计数（B7）----------
    @property
    def n_t(self) -> int:
        """N_T = 本任务全部试验数（含中止与被放弃），不含校验拒绝。DSR 的 N 取它（B1）。"""
        return sum(1 for r in self._rows.values() if r["status"] != REJECTED)

    @property
    def n_prop(self) -> int:
        """n_prop = 碰数据前就被校验拒掉的提议数，另计、不进 N_T。"""
        return sum(1 for r in self._rows.values() if r["status"] == REJECTED)

    @property
    def n_sent(self) -> int:
        """送检数 = 进过 V 段的候选数（不得超过登记的 n_fin）。"""
        return sum(1 for r in self._rows.values() if SEG_V in r["touched"])

    # ---------- 结论（B5）----------
    def problems(self) -> list[str]:
        """协议破坏清单（空 = 协议完好）。登记前记的行也在登记后一起核。"""
        reasons = list(self._events)
        reg = self._registration
        if reg is not None:
            if self.n_sent > reg["n_fin"]:
                reasons.append(f"送检数 {self.n_sent} 超过登记的 n_fin = {reg['n_fin']}（B5：送检数超出登记）")
            if reg.get("r_max") is not None:
                top = max((r["round"] for r in self._rows.values()), default=0)
                if top > reg["r_max"]:
                    reasons.append(f"轮数超登记：出现第 {top} 轮，登记 R_max = {reg['r_max']}（B5）")
        hit_test = [cid for cid, r in self._rows.items() if SEG_TEST in r["touched"]]
        if hit_test:
            reasons.append(f"候选 {hit_test} 的记录碰了测试段（A2：测试段只有考试脚本读）")
        return reasons

    @property
    def verdict(self) -> str:
        """整本账的结论：「协议完好」或「不可认证」。"""
        return UNCERTIFIABLE if self.problems() else PROTOCOL_OK

    def summary(self) -> dict:
        """B7 的导出 + 结论：N_T、n_prop、送检数、登记的 n_fin 与 R_max、登记哈希。"""
        reasons = self.problems()
        reg = self._registration or {}
        return {
            "verdict": UNCERTIFIABLE if reasons else PROTOCOL_OK, "reasons": reasons,
            "N_T": self.n_t, "n_prop": self.n_prop, "n_sent": self.n_sent,
            "n_fin": reg.get("n_fin"), "r_max": reg.get("r_max"), "registration_hash": self._reg_hash,
        }

    # ---------- 导出 ----------
    def __contains__(self, cand_id) -> bool:
        """这个候选记过账没有（断了重跑时用来跳过已记的）。"""
        return str(cand_id) in self._rows

    @staticmethod
    def _copy(row: dict) -> dict:
        out = dict(row)
        out["touched"] = tuple(row["touched"])
        out["scores"] = dict(row["scores"])
        out["s_returns"], out["v_returns"] = _returns_out(row["s_returns"]), _returns_out(row["v_returns"])
        return out

    def export_rows(self) -> list[dict]:
        """本任务的账本行（合并补记后的副本，改它不影响账本）；行数 = N_T + n_prop。"""
        return [self._copy(r) for r in self._rows.values()]

    def to_frame(self) -> pd.DataFrame:
        """账本行转成 DataFrame（列序同 ROW_FIELDS）。"""
        return pd.DataFrame(self.export_rows(), columns=list(ROW_FIELDS))
