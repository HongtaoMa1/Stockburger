# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""stockburger.gate 单测：v007 B3 / B4 / B5 / B14 / B6 / B10 / B22 的对账数 + 固定种子的合成数据。

跑法：在仓库根目录下 `python -m pytest -q`。
"""
import math

import numpy as np
import pytest
from scipy import integrate, stats
from scipy.signal import lfilter

from stockburger import gate as g

TS, TV = 504, 252          # 合成数据：S 段 2 年、V 段 1 年（交易日）
SIGMA = 0.01               # 日收益标准差 1%


def ar1_paths(n, T, a1, seed, mu=0.0, sigma=SIGMA, burn=200):
    """n 条长 T 的 AR(1) 高斯日收益：无条件标准差 sigma、均值 mu、一阶自相关 a1，固定种子。"""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, T + burn))
    x = lfilter([1.0], [1.0, -a1], e, axis=1)[:, burn:] if a1 else e[:, burn:]
    return x * sigma * math.sqrt(1.0 - a1 * a1) + mu


def mu_for_sharpe(sr_annual, sigma=SIGMA):
    """年化 Sharpe → 日均值（植入信号用）。"""
    return sr_annual / math.sqrt(252) * sigma


def one_path(seed, a1=0.0, sr_annual=0.0):
    x = ar1_paths(1, TS + TV, a1, seed, mu=mu_for_sharpe(sr_annual))[0]
    return x[:TS], x[TS:]          # (S 段, V 段)


# ================================================================ 门槛与功效（B3 / B4）：对账数
def test_kappa_reconciliation():
    """B4：κ(1, 0.05) ≈ 1.645，κ(5, 0.05) ≈ 2.84。"""
    assert g.kappa(1, 0.05) == pytest.approx(1.645, abs=1e-3)
    assert g.kappa(1, 0.05) == pytest.approx(stats.norm.isf(0.05), rel=1e-12)
    assert g.kappa(5, 0.05) == pytest.approx(2.84, abs=5e-3)


def test_kappa_structure():
    Ns = [1, 2, 5, 10, 40, 820, 11480, 123410]
    ks = [g.kappa(N) for N in Ns]
    assert all(b >= a for a, b in zip(ks, ks[1:]))          # 门槛随 N 只升不降
    for N in Ns[1:]:
        assert g.kappa(N) >= g.kappa_bonferroni(N) - 1e-12
        assert g.kappa(N) >= g.e_max(N) + stats.norm.isf(0.05) - 1e-12


def test_e_max_close_to_exact_expectation():
    """E_max 近似式与 N 个标准正态最大值期望的数值积分只差几个百分点。"""
    assert g.e_max(1) == 0.0
    for N in (5, 40, 820):
        exact = integrate.quad(lambda x: x * N * stats.norm.pdf(x) * stats.norm.cdf(x) ** (N - 1),
                               -12, 12, limit=200)[0]
        assert abs(g.e_max(N) - exact) < 0.05


def test_sr_mde_reconciliation():
    """B11：T_val = 252 时送 1 个 ≈ 2.49、送 5 个 ≈ 3.68。"""
    assert g.sr_mde(1, 252) == pytest.approx(2.49, abs=5e-3)
    assert g.sr_mde(5, 252) == pytest.approx(3.68, abs=5e-3)
    assert g.sr_mde(1, 1260) == pytest.approx(g.sr_mde(1, 252) / math.sqrt(5), rel=1e-12)
    assert g.annual_threshold(5, 252) == pytest.approx(g.kappa(5), rel=1e-12)


def test_gate_constants_echo():
    """B3：登记后算 N_T^tr、κ、年化门槛、SR_mde，并回显每多送一个 SR_mde 升了多少。"""
    c = g.gate_constants(5, 252, 0.05, 0.2)
    assert c["n_t_tr"] == 5 and c["kappa"] == pytest.approx(g.kappa(5))
    assert c["annual_threshold"] == pytest.approx(g.annual_threshold(5, 252))
    assert c["sr_mde_by_n"][0] == pytest.approx(2.49, abs=5e-3) and c["sr_mde"] == pytest.approx(3.68, abs=5e-3)
    assert all(b > a for a, b in zip(c["sr_mde_by_n"], c["sr_mde_by_n"][1:]))
    assert "送 1 个 2.49" in c["echo"] and "送 5 个 3.68" in c["echo"] and "（+" in c["echo"]


def test_input_validation():
    with pytest.raises(ValueError):
        g.kappa(0)
    with pytest.raises(ValueError):
        g.kappa(2.5)
    with pytest.raises(ValueError):
        g.kappa(5, 1.2)
    with pytest.raises(ValueError):
        g.robust_t([0.01, np.nan, 0.02, 0.0])
    with pytest.raises(ValueError):
        g.robust_t(np.ones(10))
    with pytest.raises(ValueError):
        g.gate_statistic(np.arange(10.0), "sharpe")


# ================================================================ 检验统计量（B14）
def test_robust_t_equals_statsmodels_hc1():
    import statsmodels.api as sm

    x = ar1_paths(1, TV, 0.0, seed=1, mu=0.0005)[0]
    hc1 = sm.OLS(x, np.ones(x.size)).fit(cov_type="HC1").tvalues[0]
    assert g.robust_t(x) == pytest.approx(hc1, rel=1e-10)


def test_hac_t_matches_bartlett_formula():
    assert g.hac_bandwidth(252) == 4
    x = ar1_paths(1, TV, 0.2, seed=2, mu=0.0003)[0]
    d, T, L = x - x.mean(), x.size, 4
    lrv = d @ d / T + 2 * sum((1 - k / (L + 1)) * (d[k:] @ d[:-k]) / T for k in range(1, L + 1))
    assert g.hac_t(x) == pytest.approx(x.mean() / math.sqrt(lrv / T), rel=1e-10)


def test_lag1_autocorr_matches_statsmodels_acf():
    from statsmodels.tsa.stattools import acf

    x = ar1_paths(1, TS, 0.3, seed=3)[0]
    assert g.lag1_autocorr(x) == pytest.approx(acf(x, nlags=1, fft=False)[1], rel=1e-10)


def test_a1_test_is_the_robust_z():
    """B14 默认：z = Σd_t·d_{t−1} / √Σ(d_t·d_{t−1})²，双侧，水平 0.05。"""
    x = ar1_paths(1, TS, 0.1, seed=6)[0]
    d = x - x.mean()
    prod = d[1:] * d[:-1]
    out = g.lag1_autocorr_test(x)
    assert out["z"] == pytest.approx(prod.sum() / math.sqrt(prod @ prod), rel=1e-12)
    assert out["p_value"] == pytest.approx(2 * stats.norm.sf(abs(out["z"])), rel=1e-12)
    assert out["significant"] == (out["p_value"] < 0.05) and out["a1"] == pytest.approx(g.lag1_autocorr(x))


def test_block_bootstrap_t_reproducible_and_wider_under_positive_a1():
    pytest.importorskip("arch")                                 # 块自助 t 要 arch（可选依赖）；没装就跳过
    x = ar1_paths(1, TV, 0.5, seed=4)[0] + 0.001
    t1 = g.block_bootstrap_t(x, seed=5)
    assert t1 == g.block_bootstrap_t(x, seed=5)                  # 固定种子可复现
    assert abs(t1) < abs(g.robust_t(x))                           # 正 a₁ 下块自助 se 更宽


def test_choose_statistic_branches():
    """B14 指定单测：喂 a₁ 显著与不显著的两条序列，分别走换统计量与稳健 t 两支。"""
    significant = ar1_paths(1, TS, 0.3, seed=5)[0]
    insignificant = np.random.default_rng(0).standard_normal(TS) * SIGMA
    c1 = g.choose_statistic(significant)
    assert c1.a1_significant and c1.statistic == "hac_t"
    assert g.choose_statistic(significant, switch_to="block_bootstrap_t").statistic == "block_bootstrap_t"
    c0 = g.choose_statistic(insignificant)
    assert not c0.a1_significant and c0.statistic == "robust_t"
    with pytest.raises(ValueError):
        g.choose_statistic(significant, switch_to="robust_t")


# ================================================================ 闸门判定（B4 / B5 / B6 / B10 / B22 / U-6）
def test_gate_decision_three_values_and_fixed_sentences():
    s, v = one_path(10, sr_annual=6.0)
    passed = g.gate_decision(v, 5, s_returns=s)
    assert passed.verdict == g.PASS and passed.t_value >= passed.kappa
    s0, v0 = one_path(11)
    failed = g.gate_decision(v0, 5, s_returns=s0)
    assert failed.verdict == g.FAIL and failed.t_value < failed.kappa
    broken = g.gate_decision(v, 5, s_returns=s, violations=["送检数 6 超过登记的 n_fin = 5"])
    assert broken.verdict == g.UNCERTIFIABLE and broken.t_value is None
    for r in (passed, failed, broken):
        assert "参考，不拦人" in r.note and "参考，不拦人" in r.report()
        assert g.SCOPE_NOTE in r.report()                          # B10 认证范围固定印
        assert r.protocol_broken == (r.verdict == g.UNCERTIFIABLE)  # 只有不可认证算真的不过
    assert passed.label == "过闸（参考，不拦人）"
    assert failed.label == "未过闸（参考，不拦人）"
    assert broken.label.startswith("不可认证") and "送检数 6" in broken.report()
    assert passed.guarantee == g.PASS_NOTE and "族错误率保证不适用 / 未核" in passed.guarantee   # B22：过闸一律这么印


@pytest.mark.parametrize("reason", [
    "登记被改：原登记哈希 0123456789ab…",                 # B5 的三种协议破坏（v007.1），由各模块查出、拼进 violations
    "S 段有 3 个样本的标签窗末端不早于下一段首日 2024-10-21（A3）",
    "登记被改（B5）：任务 t2 的 V 没在日历块账本登记（B9 / B16），等于没登记就碰 V",
    "候选 ['c1'] 的记录碰了测试段（A2：测试段只有考试脚本读）",
])
def test_protocol_breaks_make_uncertifiable(reason):
    s, v = one_path(10, sr_annual=6.0)                 # 信号再强，协议破了也是不可认证
    r = g.gate_decision(v, 5, s_returns=s, violations=[reason])
    assert r.verdict == g.UNCERTIFIABLE and r.violations == [reason] and r.t_value is None


def test_reuse_note_is_only_a_note():
    """B16（v007.1）：这段 V 给过 n ≥ 2 个任务时报告加注；判定和 t 都不变，不判不可认证。"""
    s, v = one_path(10, sr_annual=6.0)
    base = g.gate_decision(v, 5, s_returns=s)
    reused = g.gate_decision(v, 5, s_returns=s, v_tasks=3)
    assert (reused.verdict, reused.t_value) == (base.verdict, base.t_value) and not reused.protocol_broken
    assert reused.reuse_note == g.REUSE_NOTE.format(n=3) == "这段 V 给过 3 个任务：闸门结论更只能当参考"
    assert "加注（B16）：这段 V 给过 3 个任务：闸门结论更只能当参考" in reused.report()
    assert base.reuse_note == "" and "加注" not in base.report()
    assert g.reuse_note(1) == g.reuse_note(None) == "" and g.reuse_note(2).startswith("这段 V 给过 2 个任务")


def test_gate_powerless_warning_is_only_a_hint():
    """B5：SR_mde 高于需求目标、或历史放不下三段 → 「闸门无力认证」，只提示、不改判定。"""
    s, v = one_path(10, sr_annual=6.0)
    base = g.gate_decision(v, 5, s_returns=s)
    warned = g.gate_decision(v, 5, s_returns=s, target_sr=2.0)     # SR_mde(5) = 3.68 > 2
    assert any("闸门无力认证" in w for w in warned.warnings)
    assert warned.verdict == base.verdict and warned.t_value == base.t_value
    assert not any("闸门无力认证" in w for w in g.gate_decision(v, 5, s_returns=s, target_sr=5.0).warnings)
    assert any("闸门无力认证" in w for w in g.gate_decision(v, 5, s_returns=s, history_fits=False).warnings)


def test_report_prints_threshold_and_a1():
    s, v = one_path(10, sr_annual=6.0)
    r = g.gate_decision(v, 5, s_returns=s)
    txt = r.report()
    for key in ("a₁ =", "κ(N_T^tr = 5", "年化门槛", "SR_mde", "协议：完好"):
        assert key in txt
    assert r.a1["source"] == "S 段" and r.a1["a1"] == pytest.approx(g.lag1_autocorr(s))


def test_missing_s_segment_falls_back_with_warning():
    _, v = one_path(10, sr_annual=6.0)
    r = g.gate_decision(v, 5)
    assert r.a1["source"].startswith("V 段")
    assert any("不合 B14" in w for w in r.warnings)


# ================================================================ 合成数据蒙特卡洛
def test_null_false_pass_rate_n1_near_alpha():
    """独立零信号、N = 1：走完整闸门（含 B14）的假过闸率 ≈ α = 0.05。"""
    X = ar1_paths(3000, TS + TV, 0.0, seed=20261114)
    rate = np.mean([g.gate_decision(x[TS:], 1, s_returns=x[:TS]).verdict == g.PASS for x in X])
    assert abs(rate - 0.05) < 0.015


def test_planted_signal_passes():
    """强植入信号（年化 Sharpe 6，N = 5，κ ≈ 2.84）几乎总能过闸。"""
    X = ar1_paths(200, TS + TV, 0.0, seed=3, mu=mu_for_sharpe(6.0))
    rate = np.mean([g.gate_decision(x[TS:], 5, s_returns=x[:TS]).verdict == g.PASS for x in X])
    assert rate >= 0.95


def test_ar1_inflates_robust_t_and_b14_reduces_it():
    """a₁ = 0.3 的零均值 AR(1)：稳健 t 假过闸率明显偏高（理论 1 − Φ(1.645/c(0.3)) ≈ 0.11）；
    按 B14 换 HAC 后显著降低（配对 McNemar）。"""
    X = ar1_paths(2000, TS + TV, 0.3, seed=7)
    kap = g.kappa(1)
    robust_only = np.array([g.robust_t(x[TS:]) >= kap for x in X])
    results = [g.gate_decision(x[TS:], 1, s_returns=x[:TS]) for x in X]
    bridged = np.array([r.verdict == g.PASS for r in results])
    assert np.mean([r.statistic == "hac_t" for r in results]) > 0.99    # S 段 a₁ 几乎都显著 → 换 HAC
    assert robust_only.mean() > 0.08
    assert bridged.mean() < robust_only.mean() - 0.02
    n01 = int(np.sum(robust_only & ~bridged))
    n10 = int(np.sum(bridged & ~robust_only))
    assert stats.binomtest(n10, n01 + n10, 0.5, alternative="less").pvalue < 1e-3


def test_ar1_b14_block_bootstrap_branch():
    """同上，B14 登记成块自助 t 时，假过闸率也低于不修自相关的稳健 t。"""
    pytest.importorskip("arch")                                 # 块自助 t 要 arch（可选依赖）；没装就跳过
    X = ar1_paths(400, TS + TV, 0.3, seed=7)
    kap = g.kappa(1)
    robust_only = np.array([g.robust_t(x[TS:]) >= kap for x in X])
    boot = np.array([g.gate_decision(x[TS:], 1, s_returns=x[:TS], switch_to="block_bootstrap_t",
                                     stat_kwargs={"n_boot": 299, "seed": i}).verdict == g.PASS
                     for i, x in enumerate(X)])
    assert boot.mean() < robust_only.mean()
    assert np.sum(robust_only & ~boot) > np.sum(boot & ~robust_only)
