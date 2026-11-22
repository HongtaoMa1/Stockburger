# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""V 段闸门：门槛、检验统计量、三值判定。

规格：验收标准 v007（v007.1）的 B3（登记后的常数与回显）、B4（门槛 κ）、B5（三值）、
B14（用哪个 t）、B6 / B10 / B22（报告固定句）、B16（一段 V 给过几个任务的加注）；默认值照 v007「开跑前要登记的默认值」。
口径（A6′，U-6）：闸门结论和各种门槛照算照报、只作参考，「过闸 / 未过闸」都不拦候选去模拟盘或实盘；
只有协议被破坏才出「不可认证」。[GUESS] 数（α_gate = 0.05、β_pow = 0.2、a₁ 检验水平、块长）在这里只当
函数默认值，开跑前随 B3 登记；κ、E_max、SR_mde 是公式算的量，不算猜数。

一切产出不构成投资建议。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329
TRADING_DAYS = 252

PASS = "过闸"
FAIL = "未过闸"
UNCERTIFIABLE = "不可认证"

REFERENCE_NOTE = ("参考，不拦人（U-6）：「过闸 / 未过闸」只作参考，不拦候选去模拟盘 / 实盘；"
                  "只有「不可认证」（协议被破坏）才算真的不过")
SCOPE_NOTE = ("认证范围：循环内选择偏差；不认证前视泄漏、顶层先验、V→测试段的平稳性、"
              "S 段挑选所带来并在 V 段延续的状态持续性收益")                                   # B10，固定印
PASS_NOTE = ("按协议过闸；族错误率保证不适用 / 未核（原因：保证的前提——顶层语料和人都没见过这段 V 的噪声等——"
             "真实数据上默认不成立）")                                                     # B22，过闸者固定印
REUSE_NOTE = "这段 V 给过 {n} 个任务：闸门结论更只能当参考"     # B16（v007.1）：一段 V 从第 2 个任务起加注，不判不可认证


def reuse_note(n_tasks) -> str:
    """B16（v007.1）：这段 V 给过 n 个任务（含本任务，即 BlockLedger.check_v 的 n）时的加注。
    n ≥ 2 才加注；n ≤ 1 或没数（None）返回空串。只是加注：不改闸门判定，也不判不可认证。"""
    if n_tasks is None or isinstance(n_tasks, bool):
        return ""
    n = int(n_tasks)
    return REUSE_NOTE.format(n=n) if n >= 2 else ""


# ---------------------------------------------------------------- 小工具
def _check_alpha(alpha: float, name: str = "α") -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"{name} 必须在 (0, 1) 里，收到 {alpha}")
    return float(alpha)


def _check_n(N) -> int:
    if N < 1 or int(N) != N:
        raise ValueError(f"N 必须是 ≥ 1 的整数，收到 {N}")
    return int(N)


def _as_1d(returns) -> np.ndarray:
    x = np.asarray(returns, dtype=float).ravel()
    if x.size < 3:
        raise ValueError("收益序列太短（至少 3 天）")
    if not np.all(np.isfinite(x)):
        raise ValueError("收益里有 NaN / inf，先去掉缺值再进闸门")
    if np.ptp(x) == 0.0:
        raise ValueError("收益序列是常数，方差为 0，检验统计量无定义")
    return x


# ---------------------------------------------------------------- 门槛与功效（B3 / B4）
def e_max(N: int) -> float:
    """N 个纯噪声候选里最好那个的期望 z 值（B4）：E_max(N) = (1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))，γ 为欧拉常数。
    N = 1 时近似式发散，按定义取 0。用上侧分位 isf 算，免得 N 很大时 1−1/N 浮点坍缩。"""
    N = _check_n(N)
    if N == 1:
        return 0.0
    return float((1 - EULER_GAMMA) * norm.isf(1.0 / N)
                 + EULER_GAMMA * norm.isf(1.0 / (N * math.e)))


def kappa_bonferroni(N: int, alpha: float = 0.05) -> float:
    """Bonferroni 门槛 Φ⁻¹(1−α/N)：N 次检验各用 α/N（κ 的后一项，B4）。"""
    return float(norm.isf(_check_alpha(alpha) / _check_n(N)))


def kappa(N: int, alpha: float = 0.05) -> float:
    """闸门门槛 κ(N, α) = max{E_max(N) + Φ⁻¹(1−α), Φ⁻¹(1−α/N)}；κ(1, α) = Φ⁻¹(1−α)（B4）。
    N 取 N_T^tr（L = 0 时就是送检数 n_fin），α 取 α_gate。对账：κ(1, 0.05) ≈ 1.645，κ(5, 0.05) ≈ 2.84。"""
    N, alpha = _check_n(N), _check_alpha(alpha)
    if N == 1:
        return float(norm.isf(alpha))
    return float(max(e_max(N) + norm.isf(alpha), kappa_bonferroni(N, alpha)))


def sr_mde(N: int, T_val: int = TRADING_DAYS, alpha: float = 0.05, beta: float = 0.2) -> float:
    """这段 V 有 1 − β_pow 把握认出的最低年化 Sharpe（B3）：SR_mde = [κ(N, α) + Φ⁻¹(1−β)]·√(252/T_val)。
    对账：T_val = 252 时送 1 个 ≈ 2.49、送 5 个 ≈ 3.68——送得越多门槛越高，所以先宽筛、再少送（B11）。"""
    if T_val < 1:
        raise ValueError("T_val 必须 ≥ 1")
    _check_alpha(beta, "β_pow")
    return float((kappa(N, alpha) + norm.isf(beta)) * math.sqrt(TRADING_DAYS / T_val))


def annual_threshold(N: int, T_val: int = TRADING_DAYS, alpha: float = 0.05) -> float:
    """年化门槛 = κ(N, α)·√(252/T_val)（B3，常数区）：V 段实现的年化 Sharpe 大约要到这个数才过闸。"""
    if T_val < 1:
        raise ValueError("T_val 必须 ≥ 1")
    return float(kappa(N, alpha) * math.sqrt(TRADING_DAYS / T_val))


def gate_constants(n_fin: int, T_val: int = TRADING_DAYS, alpha_gate: float = 0.05, beta_pow: float = 0.2) -> dict:
    """B3：登记后脚本据此算的常数——N_T^tr（L = 0 时就是 n_fin）、κ、年化门槛、SR_mde，
    外加「每多送一个，SR_mde 升了多少」的回显：sr_mde_by_n[k−1] = 送 k 个时的 SR_mde，echo 是一行人话。
    前四个键与 summary.Constants 的同名字段对得上。"""
    n = _check_n(n_fin)
    by_n = [sr_mde(k, T_val, alpha_gate, beta_pow) for k in range(1, n + 1)]
    parts = [f"送 1 个 {by_n[0]:.2f}"] + [f"送 {k + 1} 个 {by_n[k]:.2f}（+{by_n[k] - by_n[k - 1]:.2f}）"
                                          for k in range(1, n)]
    return {"n_t_tr": n, "kappa": kappa(n, alpha_gate), "annual_threshold": annual_threshold(n, T_val, alpha_gate),
            "sr_mde": by_n[-1], "sr_mde_by_n": by_n,
            "echo": f"SR_mde（T_val = {T_val}，β_pow = {beta_pow:g}）：" + "；".join(parts)}


# ---------------------------------------------------------------- 检验统计量（B14）
def robust_t(returns) -> float:
    """V 段日收益均值的异方差稳健 t：t = x̄ / √(s²/T)，s² 取 ddof = 1。
    只对常数回归时它与 White HC1 稳健 t 恒等（HC1 方差 = Σe²/(T(T−1)) = s²/T），单测里对过 statsmodels。"""
    x = _as_1d(returns)
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(x.size)))


def lag1_autocorr(returns) -> float:
    """一阶自相关 a₁（今天和昨天收益的相关），样本 ACF 口径：去均值后 Σd_t·d_{t−1} / Σd_t²。"""
    x = _as_1d(returns)
    d = x - x.mean()
    return float(np.dot(d[1:], d[:-1]) / np.dot(d, d))


def lag1_autocorr_test(returns, alpha_a1: float = 0.05) -> dict:
    """a₁ 的双侧检验（B14，默认照登记）：异方差稳健 z = Σd_t·d_{t−1} / √Σ(d_t·d_{t−1})²（d 为去均值的日收益）。
    日收益有波动聚集时，普通的 z = a₁·√T 会多报显著。返回 a1、z、p_value、significant、alpha_a1。"""
    x = _as_1d(returns)
    _check_alpha(alpha_a1, "a₁ 检验水平")
    d = x - x.mean()
    prod = d[1:] * d[:-1]
    z = float(prod.sum() / math.sqrt(np.dot(prod, prod)))
    p = float(2.0 * norm.sf(abs(z)))
    return {"a1": float(np.dot(d[1:], d[:-1]) / np.dot(d, d)), "z": z, "p_value": p,
            "significant": p < alpha_a1, "alpha_a1": float(alpha_a1)}


def hac_bandwidth(T: int) -> int:
    """Newey–West 带宽 ⌊4(T/100)^{2/9}⌋，T = 252 时为 4（B14 默认）。"""
    return int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def hac_t(returns, maxlags: int | None = None) -> float:
    """均值的 HAC t（Newey–West：Bartlett 核、不做小样本修正），用 statsmodels 算（B14）。
    带宽默认 hac_bandwidth(T)。a₁ ≈ 0 时它在有限样本里反而比稳健 t 松，所以只在 a₁ 显著时换。"""
    import statsmodels.api as sm

    x = _as_1d(returns)
    lags = hac_bandwidth(x.size) if maxlags is None else int(maxlags)
    res = sm.OLS(x, np.ones(x.size)).fit(cov_type="HAC",
                                         cov_kwds={"maxlags": lags, "use_correction": False})
    return float(np.asarray(res.tvalues)[0])


def block_bootstrap_t(returns, mean_block: float = 21, n_boot: int = 999, seed=0) -> float:
    """均值的块自助 t = x̄ / se_SB（B14 的另一支）：se_SB 是平稳块自助（arch 的 StationaryBootstrap，块长服从
    几何分布、均值 mean_block）重抽 n_boot 次样本均值的标准差。改用这支要登记块长与重抽次数（默认 21、999）。
    固定 seed 可复现。"""
    from arch.bootstrap import StationaryBootstrap

    x = _as_1d(returns)
    bs = StationaryBootstrap(mean_block, x, seed=seed)
    means = np.asarray(bs.apply(np.mean, int(n_boot))).ravel()
    return float(x.mean() / means.std(ddof=1))


STATISTICS = {"robust_t": robust_t, "hac_t": hac_t, "block_bootstrap_t": block_bootstrap_t}
STAT_LABELS = {"robust_t": "异方差稳健 t", "hac_t": "HAC t（Newey–West）",
               "block_bootstrap_t": "块自助 t（平稳块自助）"}
_STAT_ALIASES = {"robust": "robust_t", "hac": "hac_t", "block_bootstrap": "block_bootstrap_t",
                 "bootstrap": "block_bootstrap_t"}


def _normalize_stat(name: str) -> str:
    name = _STAT_ALIASES.get(name, name)
    if name not in STATISTICS:
        raise ValueError(f"未知的统计量 {name!r}（可选 {sorted(STATISTICS)}）")
    return name


def gate_statistic(returns, name: str = "robust_t", **kwargs) -> float:
    """按名字算闸门统计量：robust_t / hac_t / block_bootstrap_t。"""
    return STATISTICS[_normalize_stat(name)](returns, **kwargs)


@dataclass(frozen=True)
class StatisticChoice:
    """B14 的结果：用哪个统计量、凭的 a₁ 检验。"""
    statistic: str
    a1: float
    a1_z: float
    a1_p_value: float
    a1_significant: bool
    a1_alpha: float
    a1_source: str
    reason: str


def choose_statistic(s_returns, *, switch_to: str = "hac_t", a1_alpha: float = 0.05,
                     a1_source: str = "S 段") -> StatisticChoice:
    """用哪个 t（B14，硬）：进闸门前在 S 段测送检候选日收益的 a₁——不显著用异方差稳健 t；显著就换
    HAC t 或块自助 t（switch_to，开跑前随 B3 登记），不一律换。要挡的是旧价、非同步成交、封板造出的正自相关
    把普通 t 撑大；换 t 只修尺度，不分来源。"""
    target = _normalize_stat(switch_to)
    if target == "robust_t":
        raise ValueError("switch_to 必须是 HAC 或块自助统计量（B14）")
    test = lag1_autocorr_test(s_returns, alpha_a1=a1_alpha)
    if test["significant"]:
        name = target
        reason = (f"{a1_source} a₁ = {test['a1']:.3f} 显著（p = {test['p_value']:.3g} < {a1_alpha}），"
                  f"按 B14 换成 {STAT_LABELS[name]}")
    else:
        name = "robust_t"
        reason = f"{a1_source} a₁ = {test['a1']:.3f} 不显著（p = {test['p_value']:.3g}），按 B14 用异方差稳健 t"
    return StatisticChoice(statistic=name, a1=test["a1"], a1_z=test["z"], a1_p_value=test["p_value"],
                           a1_significant=test["significant"], a1_alpha=float(a1_alpha),
                           a1_source=a1_source, reason=reason)


# ---------------------------------------------------------------- 闸门判定（B4 / B5 / B6 / B10 / B22）
@dataclass
class GateResult:
    """闸门输出。verdict 三值；只有「不可认证」算真的不过（U-6）。"""
    verdict: str
    statistic: str
    t_value: float | None
    kappa: float
    n_tests: int
    alpha: float
    T_val: int
    annual_threshold: float
    sr_mde: float
    beta: float
    target_sr: float | None
    a1: dict
    stat_reason: str
    violations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    guarantee: str = ""
    scope: str = SCOPE_NOTE
    note: str = REFERENCE_NOTE
    v_tasks: int | None = 1       # B16（v007.1）：这段 V 给过几个任务（含本任务）；≥ 2 时报告加注

    @property
    def protocol_broken(self) -> bool:
        """协议被破坏 = 真的不过（U-6）；「未过闸」不算。"""
        return self.verdict == UNCERTIFIABLE

    @property
    def reuse_note(self) -> str:
        """B16（v007.1）：这段 V 给过 n ≥ 2 个任务时的加注「这段 V 给过 n 个任务：闸门结论更只能当参考」；否则空串。"""
        return reuse_note(self.v_tasks)

    @property
    def label(self) -> str:
        """按 U-6 口径命名的判定：过闸 / 未过闸 带「参考，不拦人」，不可认证 = 真的不过。"""
        if self.verdict == UNCERTIFIABLE:
            return "不可认证（协议被破坏，真的不过）"
        return f"{self.verdict}（参考，不拦人）"

    def report(self) -> str:
        """给人读的闸门报告（中文）。"""
        a = self.a1
        t_txt = "未计算（协议被破坏，不读 V 段）" if self.t_value is None else f"{self.t_value:.3f}"
        tgt = "无" if self.target_sr is None else f"{self.target_sr:.2f}"
        lines = [
            f"闸门结果：{self.label}",
            f"  {self.note}",
            f"  统计量：{STAT_LABELS[self.statistic]}（{self.stat_reason}）",
            f"  t = {t_txt}；门槛 κ(N_T^tr = {self.n_tests}, α = {self.alpha:g}) = {self.kappa:.3f}；"
            f"年化门槛 = {self.annual_threshold:.2f}（T_val = {self.T_val}）",
            f"  SR_mde = {self.sr_mde:.2f}（β_pow = {self.beta:g}）；需求目标 = {tgt}",
            f"  a₁ = {a['a1']:.3f}（{a['source']}，稳健 z 检验 p = {a['p_value']:.3g}，"
            f"{'显著' if a['significant'] else '不显著'}）",
            f"  保证：{self.guarantee}",
            f"  {self.scope}",
            "  协议：完好" if not self.violations else "  协议破坏：" + "；".join(self.violations),
        ]
        if self.reuse_note:
            lines.append(f"  加注（B16）：{self.reuse_note}")
        lines.extend(f"  警告：{w}" for w in self.warnings)
        return "\n".join(lines)


def gate_decision(v_returns, n_tests: int, alpha: float = 0.05, *, s_returns=None, beta: float = 0.2,
                  target_sr: float | None = None, switch_to: str = "hac_t", a1_alpha: float = 0.05,
                  violations=(), history_fits: bool = True, stat_kwargs: dict | None = None,
                  v_tasks: int | None = 1) -> GateResult:
    """V 段闸门判定（B4 / B5 / B14 / B6 / B10 / B22 / B16）。

    1. 协议（B5）：violations 是各处查出来的协议破坏——TrialRegistry.problems()、SegmentSplit.purge_check、
       BlockLedger.check_v 的 reasons 拼在一起；非空就「不可认证」，不算 V 段统计量；
    2. 用哪个 t（B14）：在 S 段收益 s_returns 上测 a₁ 定统计量；没给 S 段就退到 V 段测，挂警告；
    3. 过闸 ⇔ t ≥ κ(N_T^tr, α_gate)（B4）；n_tests 即 N_T^tr（L = 0 时 = n_fin），alpha 即 α_gate；
    4. 「闸门无力认证」只提示（B5）：SR_mde 高于需求目标 target_sr，或历史放不下三段（history_fits=False）；
    5. 固定句：认证范围（B10）、过闸者印「族错误率保证不适用 / 未核」（B22）、「参考，不拦人」（U-6）；
    6. 复用（B16，v007.1）：v_tasks = 这段 V 给过几个任务（含本任务，BlockLedger.check_v 的 n）；≥ 2 时报告加注
       「这段 V 给过 n 个任务：闸门结论更只能当参考」，不改判定、不判不可认证。
    """
    N, alpha = _check_n(n_tests), _check_alpha(alpha)
    v = _as_1d(v_returns)
    T_val = int(v.size)
    warnings: list = []

    if s_returns is None:
        choice = choose_statistic(v, switch_to=switch_to, a1_alpha=a1_alpha, a1_source="V 段（S 段未提供）")
        warnings.append("a₁ 在 V 段测：没给 S 段收益，不合 B14（应在 S 段测）")
    else:
        choice = choose_statistic(s_returns, switch_to=switch_to, a1_alpha=a1_alpha, a1_source="S 段")
    a1_info = {"a1": choice.a1, "z": choice.a1_z, "p_value": choice.a1_p_value,
               "significant": choice.a1_significant, "source": choice.a1_source}

    mde = sr_mde(N, T_val, alpha, beta)
    if target_sr is not None and mde > target_sr:
        warnings.append(f"闸门无力认证：SR_mde = {mde:.2f} 高于需求目标 {target_sr:.2f}（只提示，任务照跑；B5）")
    if not history_fits:
        warnings.append("闸门无力认证：历史放不下三段（只提示，任务照跑；B5）")

    kap = kappa(N, alpha)
    violations = [str(x) for x in violations]
    if violations:
        verdict, t_val = UNCERTIFIABLE, None
        guarantee = "协议被破坏：不可认证（真的不过，U-6）"
    else:
        t_val = gate_statistic(v, choice.statistic, **(stat_kwargs or {}))
        verdict = PASS if t_val >= kap else FAIL
        guarantee = PASS_NOTE if verdict == PASS else "未过闸：无认证主张（参考，不拦人）"
    return GateResult(verdict=verdict, statistic=choice.statistic, t_value=t_val, kappa=kap,
                      n_tests=N, alpha=alpha, T_val=T_val,
                      annual_threshold=annual_threshold(N, T_val, alpha), sr_mde=mde, beta=beta,
                      target_sr=target_sr, a1=a1_info, stat_reason=choice.reason,
                      violations=violations, warnings=warnings, guarantee=guarantee, v_tasks=v_tasks)


__all__ = [
    "GateResult", "StatisticChoice", "PASS", "FAIL", "UNCERTIFIABLE", "REFERENCE_NOTE", "SCOPE_NOTE", "PASS_NOTE",
    "REUSE_NOTE", "reuse_note",
    "TRADING_DAYS", "e_max", "kappa", "kappa_bonferroni", "sr_mde", "annual_threshold", "gate_constants",
    "robust_t", "lag1_autocorr", "lag1_autocorr_test", "hac_bandwidth", "hac_t", "block_bootstrap_t",
    "STATISTICS", "STAT_LABELS", "gate_statistic", "choose_statistic", "gate_decision",
]
