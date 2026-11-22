# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""合成市场（单测与演示的夹具，不是验收标准里的条款）：N 只资产 × 约 1500 个交易日，候选特征分三类——纯噪声、植入、旧价。

不是任何真实市场，只为在没接真实数据时把流水线从头到尾跑一遍、看各条规矩起作用。一切产出不构成投资建议。

收益模型（第 t 日、第 i 只资产的日收益，按「观测价」算）：
    r[t, i] = m[t] + x_i·u[t] + b·s[t−H, i] + ε[t, i]
- m：大盘，所有资产一样；截面多空组合（权重和为 0）里正好抵消。
- u：「旧价组」（x_i = 1 的那 n_stale 只）共有的一条 AR(1) 成分，系数 φ、均值 0。它模拟价格记录被平滑：
  成交稀、常封板的股票，今天的报价只反映一部分新消息，剩下的后几天才补上，于是这组股票的收益一阶自相关为正。
- s：植入信号，第 t 日收盘后已知，决定第 t+H 日的收益。H = 标签期，默认 2：第 t 日收盘看到特征、
  第 t+1 日收盘买入、第 t+2 日收盘卖出（同 Qlib 默认标签 Ref($close,-2)/Ref($close,-1)−1）。
- ε：个股噪声。

候选特征（数组第 t 行 = 第 t 日收盘后已知的值）：
- 噪声 × n_noise：独立标准正态，和收益毫无关系；
- 植入：f = s，真有预测力（默认强度：截面多空的年化 Sharpe 约 4.5）；
- 旧价：f = x_i + 小噪声，也就是「这只股票是不是旧价组」。按它做多旧价组、做空其余，策略收益的主要成分就是 u：
  均值 0、一阶自相关明显为正（默认约 0.5）。回测里偶尔有一段看着显著，实盘做不到——正是 v007 B14
  要挡的那类正自相关（旧价、非同步成交、封板），闸门换 HAC t 或块自助 t 来修尺度。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import lfilter

NOISE, PLANTED, STALE = "噪声", "植入", "旧价"


@dataclass(frozen=True, eq=False)
class SyntheticMarket:
    """一份合成市场：交易日、资产、日收益、候选特征与各自的类别。"""
    dates: pd.DatetimeIndex
    assets: tuple[str, ...]
    returns: np.ndarray                 # (T, n) 日收益
    features: dict[str, np.ndarray]     # 特征名 → (T, n)，第 t 行 = 第 t 日收盘后已知
    kinds: dict[str, str]               # 特征名 → 噪声 / 植入 / 旧价
    label_horizon: int                  # 标签期 H
    stale_assets: tuple[str, ...]       # 旧价组
    params: dict = field(default_factory=dict)

    @property
    def label(self) -> np.ndarray:
        """标签：label[t] = returns[t + H]（第 t 日的特征对应第 t+H 日的收益）；末尾 H 行没有标签，记 NaN。"""
        h = self.label_horizon
        out = np.full_like(self.returns, np.nan)
        out[:-h] = self.returns[h:]
        return out

    def names(self, kind: str | None = None) -> list[str]:
        """候选特征名（按生成顺序）；给 kind 就只列这一类。"""
        return [k for k, v in self.kinds.items() if kind is None or v == kind]


def make_market(seed: int = 0, *, n_assets: int = 30, n_days: int = 1500, n_noise: int = 20,
                n_stale: int = 10, label_horizon: int = 2, end: str = "2026-11-13",
                sigma_m: float = 0.01, sigma_eps: float = 0.02, b_plant: float = 0.0012,
                phi_stale: float = 0.8, sigma_stale: float = 0.012,
                stale_feature_noise: float = 0.3) -> SyntheticMarket:
    """用固定种子造一份合成市场（收益模型见模块说明）。
    同一组参数、同一个 seed，结果逐位相同。

    默认数是演示用的 [GUESS]：b_plant = 0.0012 让植入特征的截面多空年化 Sharpe 约 4.5（V 段 t 约 4.5，
    多数世界稳稳越过 κ(5) ≈ 2.84）；φ = 0.8、sigma_stale = 0.012 让旧价策略收益的一阶自相关约 0.5。
    交易日用工作日历（不剔节假日），末日 end。
    """
    if n_days <= label_horizon + 2 or n_assets < 4 or not 0 < n_stale < n_assets:
        raise ValueError("资产数、交易日数或旧价组大小不合理")
    if not 0 <= phi_stale < 1:
        raise ValueError("φ 必须在 [0, 1) 里")
    rng = np.random.default_rng(seed)
    T, n, h = int(n_days), int(n_assets), int(label_horizon)

    stale_idx = np.sort(rng.choice(n, size=n_stale, replace=False))
    x = np.zeros(n)
    x[stale_idx] = 1.0
    m = rng.normal(0.0, sigma_m, T)
    burn = 200   # AR(1) 先空转 200 天，让 u 从平稳分布出发
    e = rng.normal(0.0, sigma_stale * np.sqrt(1.0 - phi_stale ** 2), T + burn)
    u = lfilter([1.0], [1.0, -phi_stale], e)[burn:]
    s = rng.standard_normal((T, n))
    eps = rng.normal(0.0, sigma_eps, (T, n))

    r = m[:, None] + x[None, :] * u[:, None] + eps
    r[h:] += b_plant * s[:-h]

    features: dict[str, np.ndarray] = {}
    kinds: dict[str, str] = {}
    width = max(2, len(str(n_noise)))
    noise = rng.standard_normal((n_noise, T, n))
    for k in range(n_noise):
        name = f"{NOISE}{k + 1:0{width}d}"
        features[name], kinds[name] = noise[k], NOISE
    features[PLANTED], kinds[PLANTED] = s, PLANTED
    features[STALE] = x[None, :] + stale_feature_noise * rng.standard_normal((T, n))
    kinds[STALE] = STALE

    assets = tuple(f"资产{i + 1:02d}" for i in range(n))
    params = dict(seed=seed, n_assets=n, n_days=T, n_noise=n_noise, n_stale=n_stale, label_horizon=h,
                  sigma_m=sigma_m, sigma_eps=sigma_eps, b_plant=b_plant, phi_stale=phi_stale,
                  sigma_stale=sigma_stale, stale_feature_noise=stale_feature_noise)
    return SyntheticMarket(
        dates=pd.bdate_range(end=end, periods=T), assets=assets, returns=r, features=features,
        kinds=kinds, label_horizon=h, stale_assets=tuple(assets[i] for i in stale_idx), params=params,
    )
