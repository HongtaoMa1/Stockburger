# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""成品：冻结、冻结即可用、影子与日报（v007 第六节 C3 / B25 / C4 / C5）。

一句话：送考的那份模型就是要冻结的那份。冻结时记下冻结时刻和三份哈希（权重、特征清单、顶层提案清单），从这一刻起成品
就「可用」；考试成绩和以后加的任何认证都只是贴在它身上的标签，可用性字段永远不变。冻结后由独立定时脚本每天用冻结权重在
新行情上推理、记分（影子），日报把测试段成绩、影子成绩、已跑天数并排印给人看；t 只显示，不设过线判定。

规格：验收标准 v007 第六节 C3 / B25 / C4 / C5，记分口径照第三节 B7。
- 冻结（C3）：默认用 S 段训的那份；想按同一配置在 S + V 上重训再考，要在碰测试段前定（trained_on="S+V"）。冻结后再训
  （含考完拿测试段数据重训，trained_on="S+V+测试"）= 新产物：哈希变了，照样可用，但日报里不挂原来那次考试成绩
  （靠指纹对不上实现）。送考那份不用测试段训，靠自觉，不设机器验法（A2 / H8）。
- 记分口径（B7）：每天按预测分数在当日截面做多空——名次去均值，多头合计 +1、空头合计 −1；每只股票取标签那段的实际收益
  （Qlib 默认 t+1 收盘到 t+2 收盘），不扣成本。逐日数直接用 screen.py 的同一套函数算；测试段成绩和影子成绩都用
  score_period 汇总：逐日截面 rank IC 的均值与 t、多空日收益的年化 IR。
- 日报里的标签照记、查表印：L_label ≠ clean 时测试段成绩旁标「压力测试 / 概念验证级」（C1）、没人确认时标「AI 代确认」、
  同一段第 2 次考起加注「在几个成绩里挑 = 在测试数据上挑」（A2）。日报只给人看，不进顶层汇总表（A7 / C6）。
v007 第十一节的前瞻证书（窗长、α_f、条件零假设、前瞻多件并发时的记账）不在这里，以后要做再加。一切产出不构成投资建议。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .records import TEST_SCORE_TAG
from .screen import daily_rank_ic, strategy_returns

TRADING_DAYS = 252
AVAILABILITY = "可用"                                  # B25：冻结即可用，任何标签都不进这个字段
TRAINED_ON = ("S", "S+V", "S+V+测试")                   # C3：前两种可以送考；第三种是考完重训的新产物
AI_CONFIRMED = "AI 代确认"                              # A2：没人时按登记的挑选规则代为确认
RESELECT_NOTE = "在几个成绩里挑 = 在测试数据上挑"          # A2：同一段第 2 次考起
NOT_ATTACHED = "冻结后重训 = 新产物：不挂这次考试成绩"      # C3
T_NOTE = "t 只显示，不设过线判定"                        # C5


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _ts(x) -> pd.Timestamp:
    return pd.Timestamp(x)


# ======================================================================
# 冻结（C3）与冻结即可用（B25）
# ======================================================================
def weights_hash(weights) -> str:
    """权重哈希（C3）。收 LightGBM 的 Booster（Qlib LGBModel 的 .model）、模型文件 Path、bytes、模型文本 str，或数值数组
    （按 float64 的形状与字节算）。"""
    if hasattr(weights, "model_to_string"):
        weights = weights.model_to_string()
    if isinstance(weights, Path):
        data = weights.read_bytes()
    elif isinstance(weights, (bytes, bytearray)):
        data = bytes(weights)
    elif isinstance(weights, str):
        data = weights.encode("utf-8")
    else:
        arr = np.asarray(weights, dtype=float)
        data = repr(arr.shape).encode("utf-8") + arr.tobytes()
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class FrozenProduct:
    """冻结的成品（C3）：冻结时刻与权重、特征清单、顶层提案清单三份哈希。哈希变了就是新产物。"""
    product_id: str
    tau_freeze: pd.Timestamp
    weights_hash: str
    features_hash: str
    proposals_hash: str
    trained_on: str = "S"

    @property
    def fingerprint(self) -> str:
        """整份成品的指纹：考试记录、日报都拿它认「是不是同一份」。

        算法（外人照这个就能重算）：sha256( json.dumps([product_id, tau_freeze 的 isoformat, weights_hash, features_hash,
        proposals_hash, trained_on], sort_keys=True, ensure_ascii=False) 的 UTF-8 字节 )——六项按这个顺序排成列表，
        json.dumps 用默认分隔符「, 」。"""
        return _sha([self.product_id, self.tau_freeze.isoformat(), self.weights_hash, self.features_hash,
                     self.proposals_hash, self.trained_on])

    @property
    def availability(self) -> str:
        """可用性（B25）：冻结即可用，恒为「可用」——不读考试成绩、认证或任何标签。"""
        return AVAILABILITY

    def same_weights(self, weights) -> bool:
        """影子脚本推理前核对：手上的模型是不是冻结的那份（C4：用冻结权重推理）。"""
        return weights_hash(weights) == self.weights_hash

    def to_json(self) -> dict:
        return {"product_id": self.product_id, "tau_freeze": self.tau_freeze.isoformat(),
                "weights_hash": self.weights_hash, "features_hash": self.features_hash,
                "proposals_hash": self.proposals_hash, "trained_on": self.trained_on,
                "fingerprint": self.fingerprint}

    @classmethod
    def from_json(cls, row: dict) -> FrozenProduct:
        p = cls(str(row["product_id"]), _ts(row["tau_freeze"]), row["weights_hash"], row["features_hash"],
                row["proposals_hash"], row.get("trained_on", "S"))
        if "fingerprint" in row and row["fingerprint"] != p.fingerprint:
            raise ValueError(f"成品 {p.product_id} 的记录对不上自己的指纹：记录被改过")
        return p


def freeze(product_id, tau_freeze, *, weights, features, proposals, trained_on: str = "S") -> FrozenProduct:
    """冻结成品（C3）：记冻结时刻与三份哈希。features / proposals 收字符串序列（按顺序算哈希）。

    trained_on：S（默认）/ S+V（在 S + V 上重训再考，要在碰测试段前定）/ S+V+测试（考完拿测试段数据重训的新产物：
    照样冻结、照样可用，只是对不上原来那份的指纹，日报不挂那次考试成绩）。"""
    if trained_on not in TRAINED_ON:
        raise ValueError(f"trained_on 只能是 {' / '.join(TRAINED_ON)}（默认 S 段训；S + V 重训要在碰测试段前定；"
                         f"考完重训记 S+V+测试），收到 {trained_on!r}")
    return FrozenProduct(product_id=str(product_id), tau_freeze=_ts(tau_freeze), weights_hash=weights_hash(weights),
                         features_hash=_sha([str(x) for x in features]),
                         proposals_hash=_sha([str(x) for x in proposals]), trained_on=trained_on)


# ======================================================================
# 记分（B7 口径）：测试段与影子共用
# ======================================================================
def _long(x, name: str) -> pd.Series:
    """统一成长表：索引 (datetime, instrument) 的 float Series。收 Qlib 的预测 / 标签（Series，或只有一列的 DataFrame；
    两层索引哪层是日期都行），也收宽表（行 = 日期、列 = 股票）。"""
    if isinstance(x, pd.DataFrame):
        if isinstance(x.index, pd.MultiIndex):
            if x.shape[1] != 1:
                raise ValueError(f"{name} 是多列的长表，只收一列（如 Qlib 的 LABEL0）")
            x = x.iloc[:, 0]
        else:
            x = x.stack(future_stack=True)
    if not isinstance(x, pd.Series) or not isinstance(x.index, pd.MultiIndex) or x.index.nlevels != 2:
        raise TypeError(f"{name} 要给索引为 (datetime, instrument) 的 pd.Series，或行 = 日期、列 = 股票的宽表")
    lv0, lv1 = x.index.get_level_values(0), x.index.get_level_values(1)
    if not pd.api.types.is_datetime64_any_dtype(lv0) and pd.api.types.is_datetime64_any_dtype(lv1):
        lv0, lv1 = lv1, lv0                                    # D.features 给的是 (instrument, datetime)
    idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(lv0), lv1], names=["datetime", "instrument"])
    return pd.Series(np.asarray(x, dtype=float), index=idx, name=name)


def _dates(s: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(s.index.get_level_values("datetime"))


def daily_scores(pred, label) -> pd.DataFrame:
    """逐日截面记分（B7）：返回按日期的表——ic（预测名次与实际收益名次的相关，即 rank IC）、ret（多空日收益）、
    n（当天预测与标签都有值的股票数）。

    直接用 screen.py 的 daily_rank_ic 与 strategy_returns（B7：账本、S 段成绩、闸门、测试段与影子都用这一个口径），
    这里只把长表摊成 (日期, 股票) 的宽表喂进去；一只股票都没有的日子不列。
    """
    p, y = _long(pred, "pred").unstack("instrument"), _long(label, "label").unstack("instrument")
    days, stocks = p.index.union(y.index), p.columns.union(y.columns)
    f = p.reindex(index=days, columns=stocks).to_numpy(dtype=float)
    r = y.reindex(index=days, columns=stocks).to_numpy(dtype=float)
    n = (np.isfinite(f) & np.isfinite(r)).sum(axis=1)
    if f.size == 0:                                            # 空表：screen 的函数要二维输入
        return pd.DataFrame({"ic": [], "ret": [], "n": []}, index=pd.DatetimeIndex([], name="datetime"))
    out = pd.DataFrame({"ic": daily_rank_ic(f, r), "ret": strategy_returns(f, r), "n": n},
                       index=pd.DatetimeIndex(days, name="datetime"))
    return out[out["n"] > 0]


def _t_of_mean(x: np.ndarray) -> float | None:
    sd = x.std(ddof=1) if x.size >= 2 else 0.0
    return float(x.mean() / (sd / math.sqrt(x.size))) if sd > 0 else None


@dataclass(frozen=True)
class PeriodScore:
    """一段时间的成绩（测试段、影子同一口径）。缺值为 None（不足两天或截面没有变化）。"""
    first: str | None          # 首个记分日
    last: str | None           # 末个记分日
    n_days: int                # 记了分的交易日数
    ic_mean: float | None      # 逐日 rank IC 的均值
    ic_t: float | None         # rank IC 均值 / 标准误（只显示，不设过线判定）
    ir: float | None           # 多空日收益的年化信息比率 mean/sd·√252

    def to_json(self) -> dict:
        return asdict(self)

    def text(self) -> str:
        def f(x, spec):
            return "—" if x is None else format(x, spec)
        span = "" if self.first is None else f"，{self.first} 至 {self.last}"
        return (f"IC 均值 {f(self.ic_mean, '.4f')}（t = {f(self.ic_t, '.2f')}）；年化 IR {f(self.ir, '.2f')}；"
                f"{self.n_days} 个交易日{span}")


def score_period(pred, label) -> PeriodScore:
    """给的预测与标签有几天就记几天（读哪段由调用方定）：rank IC 均值与 t、多空日收益的年化 IR。"""
    d = daily_scores(pred, label).dropna(subset=["ret"])
    ic = d["ic"].dropna().to_numpy()
    ret = d["ret"].to_numpy()
    sd = ret.std(ddof=1) if ret.size >= 2 else 0.0
    return PeriodScore(first=None if d.empty else d.index[0].date().isoformat(),
                       last=None if d.empty else d.index[-1].date().isoformat(),
                       n_days=int(ret.size), ic_mean=float(ic.mean()) if ic.size else None, ic_t=_t_of_mean(ic),
                       ir=float(ret.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else None)


def shadow_score(product: FrozenProduct, pred, label, as_of) -> PeriodScore:
    """影子成绩（C4）：只记冻结时刻之后、as_of 及之前的日子——冻结前的行情一天也不进来。

    pred 是定时脚本用冻结权重（先 product.same_weights 核对）在新行情上推出的分数，label 是已实现的标签收益（没实现的留空）。
    """
    p, y = _long(pred, "pred"), _long(label, "label")
    t0, t1 = product.tau_freeze, _ts(as_of)
    return score_period(p[(_dates(p) > t0) & (_dates(p) <= t1)], y[(_dates(y) > t0) & (_dates(y) <= t1)])


# ======================================================================
# 测试段成绩与日报（C5）
# ======================================================================
@dataclass(frozen=True)
class ExamRecord:
    """测试段考一次的成绩（A2）：考的是哪一份冻结成品（指纹）、哪一段、成绩，以及日报要照印的两个标记。"""
    fingerprint: str
    segment: tuple[str, str]       # 测试段 [t_1, t_2]
    score: PeriodScore
    ai_confirmed: bool = False     # 没人时按碰 V 前登记的挑选规则代为确认
    nth_on_segment: int = 1        # 同一段第几次被考（区间相交就算同一段）

    def to_json(self) -> dict:
        return {"fingerprint": self.fingerprint, "segment": list(self.segment), "score": self.score.to_json(),
                "ai_confirmed": self.ai_confirmed, "nth_on_segment": self.nth_on_segment}

    @classmethod
    def from_json(cls, row: dict) -> ExamRecord:
        return cls(row["fingerprint"], tuple(row["segment"]), PeriodScore(**row["score"]),
                   bool(row.get("ai_confirmed", False)), int(row.get("nth_on_segment", 1)))


def exam_record(product: FrozenProduct, pred, label, *, segment, ai_confirmed: bool = False,
                nth_on_segment: int = 1) -> ExamRecord:
    """记测试段成绩：冻结成品的预测在测试段 [t_1, t_2] 上记分（段外的日子不算），记下考的是哪一份。

    测试段只有考试脚本读、每条需求考一次（A2）——那是考试脚本和日历块账本的事，这里只管记分。
    """
    t1, t2 = _ts(segment[0]), _ts(segment[1])
    if t1 > t2:
        raise ValueError(f"测试段首日 {t1.date()} 晚于末日 {t2.date()}")
    if isinstance(nth_on_segment, bool) or int(nth_on_segment) != nth_on_segment or nth_on_segment < 1:
        raise ValueError(f"nth_on_segment 要是正整数，收到 {nth_on_segment!r}")
    p, y = _long(pred, "pred"), _long(label, "label")
    score = score_period(p[(_dates(p) >= t1) & (_dates(p) <= t2)], y[(_dates(y) >= t1) & (_dates(y) <= t2)])
    return ExamRecord(product.fingerprint, (t1.date().isoformat(), t2.date().isoformat()), score,
                      bool(ai_confirmed), int(nth_on_segment))


def _exam_line(product: FrozenProduct, exam: ExamRecord | None, L_label: str | None) -> str:
    if exam is None:
        return "测试段成绩：未考"
    if exam.fingerprint != product.fingerprint:
        return f"测试段成绩：{NOT_ATTACHED}"
    tags = (TEST_SCORE_TAG[L_label], AI_CONFIRMED if exam.ai_confirmed else "",
            RESELECT_NOTE if exam.nth_on_segment >= 2 else "")
    marks = "".join(f"［{t}］" for t in tags if t)
    return f"测试段成绩（{exam.segment[0]} 至 {exam.segment[1]}）：{exam.score.text()}{marks}"


def daily_report(product: FrozenProduct, shadow: PeriodScore, *, as_of, exam: ExamRecord | None = None,
                 L_label: str | None = None, extra_lines=()) -> dict:
    """日报（C5）：并排印测试段成绩、影子成绩、已跑天数；t 只显示，不设过线判定。

    exam：这份成品的考试记录（重训过的新产物对不上指纹，不挂）；L_label：顶层审计账的截止日标签（records.CallRecord.L_label，
    顶层没接上就是 None）；extra_lines：别的要进日报的几行（进考场前审查结果 D11、纯噪音能碰到的成绩 D10 等），照印。
    返回的 availability 恒为「可用」（B25），不随任何标签变。
    """
    attached = exam is not None and exam.fingerprint == product.fingerprint
    lines = [f"日报 {_ts(as_of).date().isoformat()} · 成品 {product.product_id}（{product.trained_on} 段训，冻结于 "
             f"{product.tau_freeze.isoformat()}）· 可用性：{AVAILABILITY}",
             _exam_line(product, exam, L_label),
             f"影子成绩：已跑 {shadow.n_days} 个交易日；{shadow.text()}（{T_NOTE}）"]
    lines += [str(x) for x in extra_lines]
    return {"product_id": product.product_id, "fingerprint": product.fingerprint, "as_of": _ts(as_of).isoformat(),
            "availability": AVAILABILITY, "exam": exam.to_json() if attached else None, "shadow": shadow.to_json(),
            "test_tag": TEST_SCORE_TAG[L_label], "lines": lines, "text": "\n".join(lines)}


__all__ = [
    "AVAILABILITY", "TRAINED_ON", "AI_CONFIRMED", "RESELECT_NOTE", "NOT_ATTACHED", "T_NOTE",
    "FrozenProduct", "freeze", "weights_hash", "PeriodScore", "daily_scores", "score_period", "shadow_score",
    "ExamRecord", "exam_record", "daily_report",
]
