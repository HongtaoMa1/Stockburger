# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""候选收益口径、S 段成绩（DSR）与粗筛。

规格：验收标准 v007 的 B7「候选收益口径」（每天按候选的预测分数在当日截面做多空：名次去均值，
多头合计 +1、空头合计 −1，等于相对截面均值的超额；每只股票取标签那段的实际收益，不扣成本）、
B1（S 段成绩用 DSR 形式，N 取 N_T）、B11（S 段筛完一次送 n_fin 个进 V）。

输入都是 (T, n) 数组：第 t 行是第 t 日；分数（特征或模型预测）在第 t 日收盘后已知，标签是该股票标签那段的
实际收益（Qlib 默认 t+1 收盘到 t+2 收盘）。缺值（停牌、当天不在股票池、标签还没实现）记 NaN：每天只用分数和
标签都有的股票做截面。S 段只该喂模型没见过的预测（S 段里的 valid 段或 purged CV 的折外预测）。
S 段成绩只作展示：已经用来挑候选，不再有紧缩含义（B1）。一切产出不构成投资建议。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis, norm, rankdata, skew

from .gate import e_max


def _row_mean(a: np.ndarray) -> np.ndarray:
    """沿最后一维求均值，只算有值的格子；整行没值得 NaN（不出警告）。"""
    n = np.isfinite(a).sum(axis=-1, keepdims=True)
    s = np.nansum(a, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _centered_ranks(a: np.ndarray) -> np.ndarray:
    """沿最后一维排名次（并列取平均名次）再减去当行名次均值；NaN 不参与排名、原样留 NaN。"""
    r = rankdata(a, axis=-1, nan_policy="omit")
    return r - _row_mean(r)


def daily_rank_ic(feature: np.ndarray, label: np.ndarray, min_stocks: int = 3) -> np.ndarray:
    """逐日截面 rank IC（Spearman）：第 t 日分数的名次与标签的名次的相关系数，只用两边都有值的股票。
    输入都是 (T, n)；有值的股票少于 min_stocks、或截面没有变化的日子记 NaN。"""
    f, y = np.asarray(feature, float), np.asarray(label, float)
    if f.shape != y.shape or f.ndim != 2:
        raise ValueError("分数和标签都要是同形状的 (T, n) 数组")
    ok = np.isfinite(f) & np.isfinite(y)
    rf = _centered_ranks(np.where(ok, f, np.nan))
    ry = _centered_ranks(np.where(ok, y, np.nan))
    num = np.nansum(rf * ry, axis=1)
    den = np.sqrt(np.nansum(rf * rf, axis=1) * np.nansum(ry * ry, axis=1))
    good = (ok.sum(axis=1) >= min_stocks) & (den > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(good, num / np.where(den > 0, den, 1.0), np.nan)


def long_short_weights(feature: np.ndarray) -> np.ndarray:
    """截面多空权重（B7 候选收益口径）：按分数名次去均值，多头合计 +1、空头合计 −1（权重和为 0，大盘涨跌抵消）。
    分数缺值的股票权重为 0，名次只在有分数的股票里排。"""
    w = _centered_ranks(np.asarray(feature, float))
    gross = np.nansum(np.abs(w), axis=-1, keepdims=True) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(gross > 0, w / np.where(gross > 0, gross, 1.0), 0.0)
    return np.nan_to_num(w, nan=0.0)


def strategy_returns(feature: np.ndarray, label: np.ndarray, direction: int = 1) -> np.ndarray:
    """按分数做截面多空的逐日收益（B7：账本、S 段成绩、闸门的 t、测试段与影子的 IR 都用这一个口径）。
    第 t 行 = 第 t 日收盘看到分数、拿到标签那段的收益（direction = −1 时反着做）；当日截面只含分数和标签都有的
    股票，这样的股票少于 2 只的日子记 NaN。"""
    if direction not in (1, -1):
        raise ValueError("direction 只能是 +1 或 −1")
    f, y = np.asarray(feature, float), np.asarray(label, float)
    if f.shape != y.shape:
        raise ValueError("分数和标签形状要一样")
    ok = np.isfinite(f) & np.isfinite(y)
    w = long_short_weights(np.where(ok, f, np.nan))          # 截面外的股票权重为 0
    out = direction * (w * np.where(ok, y, 0.0)).sum(axis=-1)
    return np.where(ok.sum(axis=-1) >= 2, out, np.nan)


def t_of_mean(x) -> float:
    """均值除以标准误（ddof = 1），不修自相关——粗筛口径，和闸门的异方差稳健 t 同式。缺值先去掉。"""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or np.ptp(x) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(x.size)))


def deflated_sharpe(returns, n_trials: int) -> float:
    """DSR（Bailey & López de Prado 2014，B1）：按试过的候选数把 Sharpe 往下折之后，真实 Sharpe > 0 的概率。

        DSR = Φ( (SR̂ − SR_0)·√(T−1) / √(1 − γ₃·SR̂ + (γ₄ − 1)/4·SR̂²) )，SR_0 = E_max(N)/√(T−1)

    SR̂ 是日频 Sharpe，γ₃、γ₄ 是偏度与（非超额）峰度；E_max 与闸门同式（B4）；N 取 N_T，跨试验方差取
    零假设下 SR̂ 的抽样方差 1/(T−1)。N = 1 时就是 PSR(0)。缺值先去掉。
    """
    x = np.asarray(returns, float)
    x = x[np.isfinite(x)]
    if x.size < 4 or np.ptp(x) == 0:
        raise ValueError("收益序列太短或是常数，算不了 DSR")
    T = x.size
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = float(skew(x)), float(kurtosis(x, fisher=False))
    den = math.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr, 1e-12))
    return float(norm.cdf((sr * math.sqrt(T - 1) - e_max(int(n_trials))) / den))


@dataclass(frozen=True)
class ScreenRow:
    """粗筛表的一行（B11）：S 段 rank IC 的均值与 t、方向（IC 为负就反着做）、名次、送不送进 V。"""
    name: str
    ic_mean: float
    ic_t: float
    direction: int
    rank: int
    selected: bool


def coarse_screen(features: dict, label: np.ndarray, rows, n_fin: int = 5) -> list[ScreenRow]:
    """S 段粗筛（B11：S 段筛完一次性送 n_fin 个进 V，L = 0）。

    只用 rows 选中的样本日（清洗后留下的 S 段日子，A3）；每个候选算逐日 rank IC 的均值与 t（不修自相关，
    粗筛口径），按 |t| 从大到小排、取前 n_fin 个；方向取 IC 均值的符号。同分按名字排，结果确定。
    """
    rows = np.asarray(rows)
    lab = np.asarray(label, float)[rows]
    if not np.isfinite(lab).any():
        raise ValueError("选中的样本日里没有可用的标签")
    out = []
    for name, f in features.items():
        ic = daily_rank_ic(np.asarray(f, float)[rows], lab)
        ic = ic[np.isfinite(ic)]
        mean = float(ic.mean()) if ic.size else float("nan")
        out.append((name, mean, t_of_mean(ic), -1 if mean < 0 else 1))
    order = sorted(out, key=lambda r: (-abs(r[2]) if math.isfinite(r[2]) else math.inf, r[0]))
    return [ScreenRow(name, mean, t, d, k + 1, k < n_fin)
            for k, (name, mean, t, d) in enumerate(order)]
