# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""合成市场（stockburger.synth）与候选收益口径、DSR、粗筛（stockburger.screen）的单测。

合成市场的三类候选要「名副其实」：噪声没预测力；植入真有；旧价的策略收益均值 0、一阶自相关明显为正（v007 B14 要挡的那类）。
对账：rank IC 与 scipy 的 Spearman 逐日一致（缺值的股票不进当日截面）；多空权重照 v007 B7 候选收益口径；
DSR 在 N = 1 时退化为 PSR(0)（v007 B1）。
"""
import math

import numpy as np
import pytest
from scipy import stats

from stockburger import gate as g
from stockburger.screen import (coarse_screen, daily_rank_ic, deflated_sharpe, long_short_weights,
                                strategy_returns, t_of_mean)
from stockburger.synth import NOISE, PLANTED, STALE, make_market


@pytest.fixture(scope="module")
def long_market():
    """长样本（2 万个交易日，只带 3 个噪声特征），用来核三类候选的总体性质。"""
    return make_market(seed=123, n_days=20000, n_noise=3)


def _clean(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


# ---------------------------------------------------------------- 合成市场
def test_market_shapes_kinds_and_label_shift():
    mk = make_market(seed=7)
    T, n = mk.returns.shape
    assert (T, n) == (1500, 30) and len(mk.dates) == T and len(mk.assets) == n
    assert len(mk.names(NOISE)) == 20 and mk.names(PLANTED) == [PLANTED] and mk.names(STALE) == [STALE]
    assert all(f.shape == (T, n) for f in mk.features.values())
    assert len(mk.stale_assets) == 10
    lab = mk.label
    h = mk.label_horizon
    assert h == 2
    np.testing.assert_array_equal(lab[:-h], mk.returns[h:])          # label[t] = returns[t + H]
    assert np.isnan(lab[-h:]).all()
    assert str(mk.dates[-1].date()) == "2026-11-13"


def test_market_is_deterministic_per_seed():
    a, b, c = make_market(seed=3), make_market(seed=3), make_market(seed=4)
    np.testing.assert_array_equal(a.returns, b.returns)
    for k in a.features:
        np.testing.assert_array_equal(a.features[k], b.features[k])
    assert not np.allclose(a.returns, c.returns)


def test_market_rejects_bad_parameters():
    with pytest.raises(ValueError):
        make_market(n_stale=0)
    with pytest.raises(ValueError):
        make_market(phi_stale=1.0)


def test_stale_strategy_mean_zero_but_clearly_positive_a1(long_market):
    """旧价：策略收益均值 0（长窗口 HAC t 不显著），一阶自相关约 0.5（要紧的是正 a₁）。"""
    r = _clean(strategy_returns(long_market.features[STALE], long_market.label))
    a1 = g.lag1_autocorr(r)
    assert 0.40 < a1 < 0.60
    assert abs(g.hac_t(r, maxlags=60)) < 3.0
    # 正 a₁ 让不修自相关的 t 被放大：长窗口 HAC 标准误明显大于 iid 标准误
    assert abs(g.robust_t(r)) > abs(g.hac_t(r, maxlags=60))


def test_planted_has_power_and_noise_has_none(long_market):
    lab = long_market.label
    rp = _clean(strategy_returns(long_market.features[PLANTED], lab))
    sr_ann = rp.mean() / rp.std(ddof=1) * math.sqrt(252)
    assert 3.5 < sr_ann < 6.0                                       # 默认强度：年化 Sharpe 约 4–5
    assert abs(g.lag1_autocorr(rp)) < 0.05
    assert np.nanmean(daily_rank_ic(long_market.features[PLANTED], lab)) > 0.03
    for name in long_market.names(NOISE):
        rn = _clean(strategy_returns(long_market.features[name], lab))
        assert abs(rn.mean() / rn.std(ddof=1) * math.sqrt(252)) < 0.5
        assert abs(g.lag1_autocorr(rn)) < 0.05


# ---------------------------------------------------------------- rank IC 与多空权重
def test_daily_rank_ic_matches_scipy_spearman_with_ties():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 4, size=(40, 12)).astype(float)    # 故意造并列名次
    y = rng.normal(size=(40, 12))
    y[5] = np.nan                                           # 标签缺失的日子记 NaN
    ic = daily_rank_ic(f, y)
    assert np.isnan(ic[5])
    for t in range(40):
        if t == 5:
            continue
        expected = stats.spearmanr(f[t], y[t]).statistic
        assert ic[t] == pytest.approx(expected, abs=1e-12)


def test_long_short_weights_are_dollar_neutral_and_monotone():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(5, 30))
    w = long_short_weights(f)
    np.testing.assert_allclose(w.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.where(w > 0, w, 0).sum(axis=1), 1.0)
    for t in range(5):
        order = np.argsort(f[t])
        assert np.all(np.diff(w[t][order]) > 0)


def test_strategy_returns_direction_and_missing_labels():
    rng = np.random.default_rng(2)
    f, y = rng.normal(size=(10, 8)), rng.normal(size=(10, 8))
    y[-1] = np.nan
    up, down = strategy_returns(f, y, 1), strategy_returns(f, y, -1)
    np.testing.assert_allclose(up[:-1], -down[:-1])
    assert np.isnan(up[-1]) and np.isnan(down[-1])
    np.testing.assert_allclose(up[:-1], (long_short_weights(f) * y).sum(axis=1)[:-1])
    with pytest.raises(ValueError):
        strategy_returns(f, y, 0)


# ---------------------------------------------------------------- 真实数据的缺值：缺的股票不进当日截面
def test_daily_rank_ic_uses_only_stocks_with_both_values():
    rng = np.random.default_rng(10)
    f, y = rng.normal(size=(30, 15)), rng.normal(size=(30, 15))
    f[3, [0, 4]] = np.nan                        # 当天两只股票没分数（停牌、不在股票池）
    y[7, 2] = np.nan                             # 当天一只股票没标签
    y[9, :13] = np.nan                           # 只剩 2 只：太少，记 NaN
    ic = daily_rank_ic(f, y)
    for t in (3, 7):
        m = np.isfinite(f[t]) & np.isfinite(y[t])
        assert ic[t] == pytest.approx(stats.spearmanr(f[t, m], y[t, m]).statistic, abs=1e-12)
    assert np.isnan(ic[9]) and np.isfinite(ic).sum() == 29


def test_strategy_returns_skip_missing_stocks_not_whole_days():
    rng = np.random.default_rng(11)
    f, y = rng.normal(size=(6, 10)), rng.normal(size=(6, 10))
    f[1, 3] = np.nan
    y[2, [0, 5]] = np.nan
    r = strategy_returns(f, y)
    for t, drop in ((1, {3}), (2, {0, 5})):
        cols = [i for i in range(10) if i not in drop]
        assert r[t] == pytest.approx((long_short_weights(f[t, cols]) * y[t, cols]).sum(), abs=1e-12)
    w = long_short_weights(f)
    assert w[1, 3] == 0.0 and w[1].sum() == pytest.approx(0.0, abs=1e-12)
    assert w[1][w[1] > 0].sum() == pytest.approx(1.0)                 # 多头合计 +1、空头合计 −1
    y[4, 1:] = np.nan                                                 # 只剩 1 只股票：做不了多空
    assert np.isnan(strategy_returns(f, y)[4]) and np.isfinite(strategy_returns(f, y)[[0, 1, 2, 3, 5]]).all()


def test_coarse_screen_handles_missing_values():
    rng = np.random.default_rng(12)
    T, n = 300, 20
    y = rng.normal(size=(T, n))
    f = y + rng.normal(scale=2.0, size=(T, n))
    f[rng.random((T, n)) < 0.1] = np.nan                              # 一成格子没分数
    y_obs = np.where(rng.random((T, n)) < 0.05, np.nan, y)            # 少量标签缺值
    out = coarse_screen({"f": f, "噪声": rng.normal(size=(T, n))}, y_obs, np.ones(T, dtype=bool), n_fin=1)
    assert out[0].name == "f" and out[0].selected and out[0].ic_t > 10 and not out[1].selected


# ---------------------------------------------------------------- DSR
def test_dsr_reduces_to_psr_when_one_trial():
    """N = 1：E_max = 0，DSR = PSR(0) = Φ(SR̂·√(T−1)/√(1 − γ₃SR̂ + (γ₄−1)/4·SR̂²))（Bailey & López de Prado 2014）。"""
    x = np.random.default_rng(3).normal(0.001, 0.01, 500)
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    psr = stats.norm.cdf(sr * math.sqrt(len(x) - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr * sr))
    assert deflated_sharpe(x, 1) == pytest.approx(psr, rel=1e-12)


def test_dsr_falls_with_more_trials_and_rises_with_sharpe():
    rng = np.random.default_rng(4)
    z = rng.normal(0, 0.01, 750)
    weak, strong = z + 0.0003, z + 0.0015
    vals = [deflated_sharpe(weak, n) for n in (1, 5, 22, 100)]
    assert all(b < a for a, b in zip(vals, vals[1:]))               # 试得越多，扣得越狠
    assert deflated_sharpe(strong, 22) > deflated_sharpe(weak, 22)
    assert 0.0 < deflated_sharpe(z, 22) < 1.0
    with pytest.raises(ValueError):
        deflated_sharpe(np.ones(10), 5)


# ---------------------------------------------------------------- 粗筛
def test_coarse_screen_picks_top_abs_t_with_direction():
    rng = np.random.default_rng(5)
    T, n = 600, 30
    y = rng.normal(size=(T, n))
    feats = {
        "强正": y + rng.normal(scale=3.0, size=(T, n)),
        "强负": -y + rng.normal(scale=3.0, size=(T, n)),
        "弱正": y + rng.normal(scale=12.0, size=(T, n)),
        "噪声a": rng.normal(size=(T, n)),
        "噪声b": rng.normal(size=(T, n)),
    }
    rows = np.ones(T, dtype=bool)
    out = coarse_screen(feats, y, rows, n_fin=2)
    assert [r.rank for r in out] == [1, 2, 3, 4, 5]
    assert {r.name for r in out if r.selected} == {"强正", "强负"}
    by = {r.name: r for r in out}
    assert by["强正"].direction == 1 and by["强负"].direction == -1
    assert by["强负"].ic_mean < 0 < by["强正"].ic_mean
    assert abs(out[0].ic_t) >= abs(out[1].ic_t) >= abs(out[2].ic_t)
    ic = daily_rank_ic(feats["弱正"], y)
    assert by["弱正"].ic_t == pytest.approx(t_of_mean(ic), rel=1e-12)


def test_coarse_screen_only_uses_selected_rows():
    rng = np.random.default_rng(6)
    T, n = 400, 20
    y = rng.normal(size=(T, n))
    f = rng.normal(size=(T, n))
    f[200:] = y[200:]                    # 后半段完全泄漏；只用前半段就看不出来
    rows = np.zeros(T, dtype=bool)
    rows[:200] = True
    first_half = coarse_screen({"f": f}, y, rows, n_fin=1)[0]
    assert abs(first_half.ic_t) < 4
    full = coarse_screen({"f": f}, y, np.ones(T, dtype=bool), n_fin=1)[0]
    assert full.ic_t > 20
