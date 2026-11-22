# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""Stockburger（股堡）原型 · 集成演示（合成数据）。

跑法（在仓库根目录下）：
    python demo.py                   # 主线 + 末尾的多世界附表（200 个世界，二十秒左右）
    python demo.py --worlds 0        # 只跑主线，一两秒
    python demo.py --seed 0          # 没挑过的世界：旧价候选换成 HAC t 后照样过闸，所以过闸只作参考（U-6）
    python demo.py --stale-mc 4500   # 再加「旧价」零分布的直接蒙特卡洛，多十几秒

规格：验收标准 v007（v007.1）；各步标的是条款号，大意见 README「条款编号对照」。
主线：造合成市场 → 切三段、清洗（A1 / A3）→ S 段粗筛、试验账本（B11 / B1 / B7）→ 碰 V 前登记（B3）→ V 段账本（V 默认紧挨
测试段；一段 V 可以给多个任务用）、用哪个 t、闸门（B19 / B9 / B16 / B14 / B4 / B5 / B6 / B10 / B22）→ 汇总表（B1）→ 确认送考（A2）→ 冻结、测试段考一次
（C3 / B25 / A2）→ 账本落状态（B16 / B7）→ 审计记录行（C1 / C2 / C13）→ 影子与日报（C4 / C5）→ 收单演示（D1–D10、D12）。
口径（A6′，U-6）：「过闸 / 未过闸」只作参考、不拦人；只有协议被破坏的「不可认证」才算真的不过；成品冻结即可用（B25）。
顶层大模型还没接：粗筛脚本顶替它挑候选，审计行用一个假设的模型。一切产出不构成投资建议。
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from stockburger import gate as G
from stockburger import intake as I
from stockburger import product as P
from stockburger import records as R
from stockburger import summary as S
from stockburger import toplayer_prompt as T
from stockburger.registry import DONE, REJECTED, TrialRegistry
from stockburger.screen import coarse_screen, deflated_sharpe, strategy_returns
from stockburger.segments import (BY_AI, BY_HUMAN, EXAMINED, SEG_S, SEG_TEST, SEG_V, BlockLedger, SegmentSplit,
                                  split_segments)
from stockburger.synth import NOISE, PLANTED, STALE, SyntheticMarket, make_market

# ---------------- 演示设定（[GUESS] 的数只能问、标、展示，A6）----------------
DEMO_SEED = 10              # 挑过的种子：从 0 往上数，第一个「三件事都看得到」的世界（见多世界附表与 README）
MARKET = "合成市场"
TASK_ID = "演示任务-001"
DEMAND_ID = "需求-001"
N_FIN = 5                   # [GUESS] B11：默认 L = 0、送检数 n_fin ≤ 5
ALPHA_GATE = 0.05           # [GUESS] B3：闸门水平 α_gate
BETA_POW = 0.2              # [GUESS] B3：β_pow = 1 − 功效
T_VAL = 252                 # [GUESS] A1：V 段长（交易日）
T_TEST = 252                # [GUESS] A1：测试段长（交易日）
R_MAX = 1                   # B3：轮数上限（演示只跑一轮）
BLOCK_FREQ = "M"            # B9：日历块长（月），开跑前登记
SWITCH_TO = "hac_t"         # B14：S 段 a₁ 显著时换成 HAC t，开跑前登记
A1_ALPHA = 0.05             # B14：a₁ 检验水平
PICK_RULE = "V 段 t 最高（基线也参加）"   # A2：没人时代为确认用的挑选规则，碰 V 前随 B3 登记
TOP_MODEL = "示例顶层模型（假设）"
TOP_CUTOFF = "2026-03"      # 假设的厂商申报截止月
DEMO_DEMAND = "日频调仓，在测试段里跑赢合成市场的等权组合"   # 第 9 步顶层提示里的需求原文（演示用）
SHADOW_SEED_SHIFT = 7919    # 冻结之后的新行情用另一个种子续造（同一套造数规则），只为演示影子
SHADOW_DAYS = (30, 250)     # 冻结后第 30 个交易日印一份日报；第 250 个交易日再印影子那一行（影子默认一直跑）
REQUESTS = (                # 收单演示：拒 / 问 / 转译 各一条（第 1、3 条是初始计划第 5 节的例子）
    "用辩证法给我做一个年化 50%、风险为 0 的策略",
    "我很保守，但想要翻倍",
    "追求高风险高收益、喜欢日内交易、关注伊朗局势",
)
BAR = "=" * 78


# ---------------- 小工具 ----------------
def _w(s: str) -> int:
    """终端显示宽度：中日韩全角字符算 2 格，组合附加符（ρ̂ 的帽子）算 0 格。"""
    return sum(0 if unicodedata.combining(ch) else 2 if unicodedata.east_asian_width(ch) in "WF" else 1
               for ch in s)


def _pad(s, width: int, right: bool = False) -> str:
    s = str(s)
    gap = " " * max(0, width - _w(s))
    return gap + s if right else s + gap


def _table(header, rows, rights=()) -> list[str]:
    cells = [list(map(str, header))] + [list(map(str, r)) for r in rows]
    widths = [max(_w(row[k]) for row in cells) for k in range(len(header))]
    return ["  " + "  ".join(_pad(c, widths[k], k in rights) for k, c in enumerate(row)).rstrip()
            for row in cells]


def _d(ts) -> str:
    return str(pd.Timestamp(ts).date())


def _sentences(text: str) -> list[str]:
    """长消息按句号、换行切成几行，方便在终端里读。"""
    return [p.strip() for p in re.split(r"(?<=。)|\n", text) if p.strip()]


# ---------------- 世界：市场 + 切分 + S / V 段的样本日 ----------------
@dataclass(frozen=True, eq=False)
class World:
    """一个合成世界（A1、A3）：市场、三段切分，以及 S / V 段清洗后留下的样本日（布尔掩码，长 T）。
    测试段只有考试那一步读（A2），这里不预先算它的任何东西。"""
    market: SyntheticMarket
    split: SegmentSplit
    label: np.ndarray
    rows_s: np.ndarray       # S 段清洗后留下的样本日（A3）
    rows_v: np.ndarray       # V 段清洗后留下的样本日


def build_world(seed: int) -> World:
    """造市场并切三段（A1）；S / V 段只留标签窗末端早于下一段首日的样本（A3，照默认一个都不删）。"""
    mk = make_market(seed=seed)
    sp = split_segments(mk.dates, t_val=T_VAL, t_test=T_TEST, label_horizon=mk.label_horizon)
    return World(mk, sp, mk.label, sp.purge_keep(SEG_S, mk.dates), sp.purge_keep(SEG_V, mk.dates))


def candidate_returns(world: World, name: str, direction: int, rows) -> pd.Series:
    """候选 name 在 rows 选中的样本日上的截面多空日收益（B7 口径），按样本日索引；方向在 S 段定，之后不改。
    只给 S 段或 V 段的样本日：测试段只有考试那一步读（A2）。"""
    mk = world.market
    r = strategy_returns(mk.features[name][rows], world.label[rows], direction)
    return pd.Series(r, index=mk.dates[rows], name=name)


def rank_by_v_t(selected, results) -> list[str]:
    """送检候选按 V 段 t 从高到低排（协议被破坏、没算 t 的排最后）；同分按名字。"""
    def key(r):
        t = results[r.name].t_value
        return (t is None, -(t or 0.0), r.name)
    return [r.name for r in sorted(selected, key=key)]


def ai_pick(selected, results) -> tuple[str, str]:
    """A2：没人时按碰 V 前登记的挑选规则代为确认——默认 V 段 t 最高的那个（基线也参加；演示里没有基线）。
    闸门结果只作参考（U-6）：没过闸的也能被挑。"""
    name = rank_by_v_t(selected, results)[0]
    t = results[name].t_value
    if t is None:
        raise RuntimeError("没有一个送检候选算出 V 段 t（协议被破坏），按规则挑不出来")
    return name, f"按登记的挑选规则「{PICK_RULE}」，「{name}」的 V 段 t = {t:.2f}，最高"


def frozen_predictions(mk: SyntheticMarket, name: str, direction: int) -> pd.DataFrame:
    """冻结成品的推理（演示版）：「模型」就是特征 name 乘方向，逐日给每只资产一个分数（行 = 日期、列 = 资产）。
    真实运行里这一步是用冻结的 LightGBM 权重在 Qlib 特征上推理。"""
    return pd.DataFrame(direction * mk.features[name], index=mk.dates, columns=list(mk.assets))


def label_frame(mk: SyntheticMarket) -> pd.DataFrame:
    """标签宽表（行 = 日期、列 = 资产）：第 t 行 = 第 t+1 日收盘到第 t+H 日收盘的收益。"""
    return pd.DataFrame(mk.label, index=mk.dates, columns=list(mk.assets))


def labels_known_by(label: pd.DataFrame, as_of, horizon: int) -> pd.DataFrame:
    """as_of 收盘时已经实现的标签：第 t 日的标签要到第 t+H 日收盘才知道，还没实现的留空（影子只记已实现的）。"""
    pos = int(label.index.searchsorted(pd.Timestamp(as_of), side="right")) - 1
    known = np.arange(len(label.index)) + horizon <= pos
    return pd.DataFrame(np.where(known[:, None], label.to_numpy(), np.nan), index=label.index, columns=label.columns)


def shadow_market(seed: int, after, n_days: int) -> SyntheticMarket:
    """冻结之后才到来的新行情（合成）：同一套造数规则另造一段市场，交易日接在 after 之后。只为演示影子（C4）。"""
    dates = pd.bdate_range(start=pd.Timestamp(after) + pd.offsets.BDay(1), periods=n_days)
    return make_market(seed=seed + SHADOW_SEED_SHIFT, n_days=n_days, end=str(dates[-1].date()))


# ---------------- 主线 ----------------
def run_pipeline(seed: int = DEMO_SEED, *, say=print, run_date=None) -> dict:
    """跑一遍主线（验收标准 v007 里已实现的部分），边跑边用 say 印给人看（测试里传一个不印的 say）。
    返回各步的关键结果。"""
    run_date = run_date or date.today()
    out: dict = {"seed": seed}

    # ======== 第 0 步：合成市场 ========
    world = build_world(seed)
    mk, sp = world.market, world.split
    names = mk.names()
    say(BAR)
    say("Stockburger（股堡）原型 · 合成数据演示（规格：验收标准 v007）")
    say("一切产出不构成投资建议。数据是现造的；顶层大模型还没接，由脚本顶替。")
    say("口径（U-6）：「过闸 / 未过闸」只作参考、不拦人；只有协议被破坏的「不可认证」才算真的不过。")
    say(BAR)
    say(f"\n【第 0 步】造一份合成市场（固定种子 {seed}）")
    say(f"  {len(mk.assets)} 只资产 × {len(mk.dates)} 个交易日（{_d(mk.dates[0])} … {_d(mk.dates[-1])}，工作日历）")
    say(f"  候选特征 {len(names)} 个：噪声 {len(mk.names(NOISE))} 个（和收益毫无关系）、植入 1 个（真有预测力）、"
        "旧价 1 个（策略收益均值 0、一阶自相关约 0.5）")
    say(f"  旧价组 {len(mk.stale_assets)} 只：{'、'.join(mk.stale_assets)}——这几只的价格记录被平滑，收益正自相关")
    say(f"  标签：第 t 日收盘看到特征 → 拿第 t+1 日收盘到第 t+{mk.label_horizon} 日收盘的收益（同 Qlib 默认标签，"
        f"标签期 {mk.label_horizon}）")

    # ======== 第 1 步：切三段、清洗 ========
    say("\n【第 1 步】切三段、清洗（A1：切分日期只能人定或取默认，顶层不能设也不能改；A3）")
    seg_names = {SEG_S: "S 段", SEG_V: "V 段", SEG_TEST: "测试段"}
    roles = {SEG_S: "训练、调参、筛选", SEG_V: "只给闸门用一次", SEG_TEST: "只给最后选中的那一个考一次"}
    for seg in (sp.s, sp.v, sp.test):
        say(f"  {_pad(seg_names[seg.name], 8)}{_d(seg.start)} … {_d(seg.end)}，{seg.n_days} 个交易日"
            f"（{roles[seg.name]}）")
    say(f"  段间空 d_pur = {sp.d_pur} 个交易日（= 标签期）；t_1 = {_d(sp.t_1)}，t_2 = 数据末日 {_d(sp.t_2)}")
    purge = sp.purge_check(sp.label_end(mk.dates[world.rows_s]), sp.label_end(mk.dates[world.rows_v]))
    n_s, n_v = int(world.rows_s.sum()), int(world.rows_v.sum())
    say(f"  清洗：标签窗伸进下一段的样本要删；段间空档不短于标签期，照默认一个都不用删 → S 段留 {n_s} 天、"
        f"V 段留 {n_v} 天；清洗检查：{purge['verdict']}")
    say(f"  切分指纹 {sp.fingerprint()[:16]}…（第 3 步登记时写死，事后比对就知道切分日期有没有人改过）")
    for w in sp.warnings:
        say(f"  警告：{w}")
    out.update(world=world, purge=purge)

    # ======== 第 2 步：S 段粗筛 + 试验账本 ========
    say(f"\n【第 2 步】S 段粗筛 + 试验账本（B11：S 段筛完一次性送 n_fin = {N_FIN} 个进 V，L = 0；B1；B7）")
    screen = coarse_screen(mk.features, world.label, world.rows_s, N_FIN)
    by_name = {r.name: r for r in screen}
    n_t = len(screen)
    s_paths, dsr = {}, {}
    reg = TrialRegistry(TASK_ID)            # 演示只记在内存里；真实运行给 path，写进 JSONL、跨任务共用
    for name in names:                      # 按试验顺序记账，一个候选一行（B7）
        row = by_name[name]
        s_paths[name] = candidate_returns(world, name, row.direction, world.rows_s)
        dsr[name] = deflated_sharpe(s_paths[name], n_t)                 # B1：DSR 形式，N 取 N_T
        reg.record(name, status=DONE, touched=(SEG_S,), proposal_text=f"特征「{name}」做截面多空，方向 {row.direction:+d}",
                   scores={"S_IC均值": row.ic_mean, "S_IC_t": row.ic_t, "S_DSR": dsr[name]}, s_returns=s_paths[name])
    reg.record("提议-前视", status=REJECTED, proposal_text="Ref($close, -2)：特征里用了第 t+2 日的收盘价")
    shown = screen[:8]
    for line in _table(["名次", "候选", "类别", "IC 均值", "IC 的 t", "方向", f"S 段 DSR（N = {n_t}）", "送 V"],
                       [[r.rank, r.name, mk.kinds[r.name], f"{r.ic_mean:+.4f}", f"{r.ic_t:+.2f}", f"{r.direction:+d}",
                         f"{dsr[r.name]:.3f}", "是" if r.selected else ""] for r in shown],
                       rights=(0, 3, 4, 6)):
        say(line)
    say(f"  ……其余 {n_t - len(shown)} 个的 |t| 都更小（全部 {n_t} 个的 DSR 见第 5 步汇总表 S 区）。"
        "「类别」只给人看，流水线不知道谁是谁。")
    say("  粗筛按逐日截面 rank IC 的 t 排；方向取 IC 的符号（IC 为负就反着做）。演示的候选就是特征本身、没有训练，"
        "整段 S 都算「模型没见过」；接 Qlib 后这里换成 valid 段或 purged CV 的折外预测（B7）")
    say(f"  试验账本：{n_t} 个候选各记一行（带 S 段逐日收益）；另有 1 个提议碰数据前就被校验拒掉（用了未来价格），"
        "计 n_prop、不进 N_T")
    out.update(screen=screen, dsr=dsr, registry=reg)

    # ======== 第 3 步：碰 V 前登记 ========
    say("\n【第 3 步】碰 V 前登记（B3：写死以后再改 = 登记被改 → 不可认证）")
    consts = G.gate_constants(N_FIN, T_VAL, ALPHA_GATE, BETA_POW)
    split_days = {k: None if v is None else [_d(v[0]), _d(v[1])] for k, v in sp.ranges().items()}
    reg_hash = reg.register(
        N_FIN, r_max=R_MAX, split=split_days, split_fingerprint=sp.fingerprint(), d_pur=sp.d_pur,
        label_horizon=sp.label_horizon, data_version=f"合成市场 make_market(seed={seed})", alpha_gate=ALPHA_GATE,
        beta_pow=BETA_POW, T_val=T_VAL, k_S=n_t, L=0,
        statistic={"a1_test": "双侧稳健 z", "a1_alpha": A1_ALPHA, "switch_to": SWITCH_TO},
        pick_rule=PICK_RULE, block_freq=BLOCK_FREQ)
    say(f"  写死：切分日期（V 段 {split_days[SEG_V][0]} … {split_days[SEG_V][1]}，测试段 {_d(sp.t_1)} … {_d(sp.t_2)}）"
        f"与指纹、d_pur = {sp.d_pur}、数据包版本")
    say(f"  α_gate = {ALPHA_GATE} [GUESS]；β_pow = {BETA_POW} [GUESS]；T_val = {T_VAL}；R_max = {R_MAX}；"
        f"k_S = {n_t}（本轮 S 段批量）；n_fin = {N_FIN}；L = 0")
    say(f"  用哪个 t（B14）：S 段 a₁ 用双侧稳健 z 检验、水平 {A1_ALPHA}；显著就换「{G.STAT_LABELS[SWITCH_TO]}」"
        f"（Bartlett 核，带宽 ⌊4(T/100)^(2/9)⌋）")
    say(f"  挑选规则（A2：没人时代为确认用）：{PICK_RULE}")
    say(f"  登记哈希 {reg_hash[:16]}…")
    say(f"  脚本据此算（B3 / B4）：N_T^tr = n_fin = {consts['n_t_tr']}（L = 0），κ = {consts['kappa']:.3f}，"
        f"年化门槛 = {consts['annual_threshold']:.2f}，SR_mde = {consts['sr_mde']:.2f}")
    say(f"  {consts['echo']}")
    say("  ——送得越多门槛越高；真实 Sharpe 低于 SR_mde 的候选「未过闸」可能只是功效不够，不等于没信号（B11）")
    out.update(reg_hash=reg_hash, consts=consts)

    # ======== 第 4 步：V 段账本登记 → 用哪个 t → 闸门 ========
    say("\n【第 4 步】V 段：账本登记 → 用哪个 t → 闸门（B19、B9 / B16、B14、B4 / B5、B6 / B10 / B22）")
    selected = [r for r in screen if r.selected]
    ledger = BlockLedger(block_freq=BLOCK_FREQ)     # 演示只记在内存里；真实运行给 path，V 段与测试段共用一本账
    acq = ledger.register_v(MARKET, sp.v.start, sp.v.end, TASK_ID)    # B19：V 就是切分里紧挨测试段的那一段，不挪
    if not acq["registered"]:
        raise RuntimeError(f"V 段登记不过：{acq['reasons']}")
    say(f"  V 段账本（B9 / B16）：V 默认紧挨测试段（B19），盖到 {len(acq['blocks'])} 个月块（{acq['blocks'][0]} … "
        f"{acq['blocks'][-1]}），登记即记「已用」；{TASK_ID} 是这段 V 的第 {acq['n']} 个任务。"
        "一段 V 可以给多个任务用：从第 2 个任务起闸门报告和汇总表加注，不判不可认证")
    for r in selected:
        reg.enter_v(r.name)                                   # 送检报到：账本上这几个候选碰了 V
    check = ledger.touch_v(TASK_ID)                           # 首次读 V 段数据之前报到，顺带查账（带复用信息）
    violations = reg.problems() + purge["reasons"] + check["reasons"]     # B5：三处查出的协议破坏拼在一起
    say(f"  读 V 前查协议（B5）：试验账本 {reg.verdict}；清洗检查 {purge['verdict']}；V 段账本 {check['verdict']}"
        + ("" if not violations else " → " + "；".join(violations)))
    v_paths, choices, results, t_robust = {}, {}, {}, {}
    for r in selected:
        v_paths[r.name] = candidate_returns(world, r.name, r.direction, world.rows_v)
        choices[r.name] = G.choose_statistic(s_paths[r.name], switch_to=SWITCH_TO, a1_alpha=A1_ALPHA)
        results[r.name] = G.gate_decision(v_paths[r.name], N_FIN, ALPHA_GATE, s_returns=s_paths[r.name],
                                          beta=BETA_POW, switch_to=SWITCH_TO, a1_alpha=A1_ALPHA,
                                          violations=violations, history_fits=not sp.warnings, v_tasks=check["n"])
        t_robust[r.name] = G.robust_t(v_paths[r.name])
    kap = consts["kappa"]
    say(f"  闸门：过闸 ⇔ V 段 t ≥ κ(N_T^tr = {N_FIN}, α_gate = {ALPHA_GATE}) = {kap:.3f}；"
        "用哪个 t 由 S 段的 a₁ 自动定（B14）")
    rows = []
    for r in selected:
        res, c = results[r.name], choices[r.name]
        note = ""
        if c.statistic != "robust_t" and t_robust[r.name] >= kap and res.verdict == G.FAIL:
            note = "← 稳健 t 越过 κ（看着显著）；a₁ 显著 → 换 HAC t 后没过：B14 拦下"
        rows.append([r.name, mk.kinds[r.name], f"{c.a1:+.3f}（p = {c.a1_p_value:.2g}）", G.STAT_LABELS[c.statistic],
                     f"{t_robust[r.name]:.2f}", "—" if res.t_value is None else f"{res.t_value:.2f}", res.label,
                     note])
    for line in _table(["候选", "类别", "S 段 a₁", "所用统计量", "稳健 t", "所用 t", "结果", ""], rows,
                       rights=(4, 5)):
        say(line)
    say("  （「稳健 t」一栏对换了统计量的候选是反事实：要是不分青红皂白一律用稳健 t，它的 t 是多少）")
    for name in (STALE, PLANTED):
        if name in results:
            say(f"\n  —— {name}候选的完整闸门报告 ——")
            for line in results[name].report().splitlines():
                say("  " + line)
    verdicts = [res.verdict for res in results.values()]
    task_result = (G.UNCERTIFIABLE if G.UNCERTIFIABLE in verdicts else G.PASS if G.PASS in verdicts else G.FAIL)
    ledger.finish_v(TASK_ID, task_result)
    for r in selected:
        reg.append_info(r.name, scores={"V_闸门": results[r.name].verdict, "V_统计量": choices[r.name].statistic},
                        v_returns=v_paths[r.name])
    say(f"\n  任务以「{task_result}」结束；V 段的块保持「已用」。送检候选的闸门结果、所用统计量和 V 段逐日收益补记进试验账本"
        "（只追加，原行不动）")
    out.update(selected=[r.name for r in selected], choices=choices, results=results, t_robust=t_robust, kappa=kap,
               ledger=ledger, v_acq=acq, v_check=check, violations=violations, task_result=task_result)

    # ======== 第 5 步：汇总表 ========
    s_consts = S.Constants(n_t_tr=consts["n_t_tr"], kappa=consts["kappa"], annual_threshold=consts["annual_threshold"],
                           sr_mde=consts["sr_mde"], rounds_left=R_MAX - 1, alpha_gate=ALPHA_GATE)
    table = S.build_summary(TASK_ID, 1, s_consts, [S.SRow(name, dsr[name]) for name in names], n_fin=N_FIN,
                            v_results=[S.VResult(r.name, results[r.name].verdict) for r in selected],
                            protocol_issues=violations, gate_warning=bool(sp.warnings), reuse_note=check["note"])
    text = S.render_summary(table)
    say("\n【第 5 步】顶层每轮读的汇总表（B1：常数区 / S 区 / V 区只给「过闸 / 未过闸」；测试段、影子、审计账一律不进）")
    for line in text.splitlines():
        say("  " + line)
    say(f"  白名单断言（A4，可选）：{'通过' if not S.check_whitelist(table) else S.check_whitelist(table)}")
    out.update(summary=table, summary_text=text)

    # ======== 第 6 步：确认送考 ========
    pick, why = ai_pick(selected, results)
    say("\n【第 6 步】确认送考（A2：测试段只收一个候选，而且要人确认过；闸门结果只作参考）")
    say(f"  演示里没人在场：{why} → 送考「{pick}」，日报标「{P.AI_CONFIRMED}」")
    say("  有人在场时由人看 S 区成绩、闸门结论和进考场前审查来挑，没过闸的也能挑（U-6）")
    out.update(pick=pick, pick_reason=why)

    # ======== 第 7 步：冻结 → 测试段考一次 ========
    say("\n【第 7 步】冻结 → 测试段考一次（C3：送考的就是要冻结的那份；B25：冻结即可用；A2：每条需求考一次）")
    direction = by_name[pick].direction
    proposal = f"特征「{pick}」做截面多空，方向 {direction:+d}"
    frozen = P.freeze(f"成品-{pick}", pd.Timestamp(sp.t_2).replace(hour=20), weights=[float(direction)],
                      features=[pick], proposals=[proposal], trained_on="S")
    say(f"  冻结：{frozen.product_id}（S 段训），冻结于 {frozen.tau_freeze}（数据末日收盘后）；权重 / 特征清单 / 提案清单的"
        f"哈希 {frozen.weights_hash[:8]}… / {frozen.features_hash[:8]}… / {frozen.proposals_hash[:8]}…")
    say(f"  可用性 = {frozen.availability}：冻结这一刻就能拿去用。演示的「模型」就是特征乘方向；之后再训（含考完拿测试段重训）"
        "= 新产物，照样可用，但日报不挂这次考试成绩")
    start = ledger.start_exam(MARKET, sp.test.start, sp.test.end, demand_id=DEMAND_ID, task_id=TASK_ID, cand_id=pick,
                              confirmed_by=BY_AI)
    if not start["allowed"]:
        raise RuntimeError(f"开考失败：{start['reasons']}")
    exam = P.exam_record(frozen, frozen_predictions(mk, pick, direction), label_frame(mk), segment=(sp.t_1, sp.t_2),
                         ai_confirmed=True, nth_on_segment=start["k"])
    say(f"  开考报到：账本给测试段的块记「已考」；本段第 {start['k']} 场，确认人「{start['confirmed_by']}」")
    say(f"  （合成数据）「{pick}」在测试段：{exam.score.text()}")
    say("  考试脚本是测试段唯一的读者；成绩只进日报（第 10 步），不写回顶层能读的地方")
    resume = ledger.start_exam(MARKET, sp.test.start, sp.test.end, demand_id=DEMAND_ID, task_id=TASK_ID, cand_id=pick,
                               confirmed_by=BY_AI)
    say(f"  断了重跑同一场（同一任务、需求、候选、区间）→ 照原来那场接着考，不算再考（resumed = {resume['resumed']}）")
    other = rank_by_v_t(selected, results)[1]
    again = ledger.start_exam(MARKET, sp.test.start, sp.test.end, demand_id=DEMAND_ID, task_id=TASK_ID, cand_id=other,
                              confirmed_by=BY_HUMAN)
    say(f"  看完成绩想换「{other}」再考同一段 → {again['verdict']}：{'；'.join(again['reasons'])}")
    out.update(frozen=frozen, exam_start=start, exam=exam, exam_resume=resume, exam_again=again)

    # ======== 第 8 步：账本落状态 ========
    say("\n【第 8 步】账本落状态（B16；B7）")
    vs = ledger.status(MARKET, sp.v.start, sp.v.end)
    ts = ledger.status(MARKET, sp.test.start, sp.test.end)
    say(f"  V 段 {len(vs['blocks'])} 个月块：{vs['state']}（给过 {'、'.join(vs['used_by'])}）")
    examined = [b for b, st in ts["block_states"].items() if st == EXAMINED]
    shared = [b for b, st in ts["block_states"].items() if st != EXAMINED]
    say(f"  测试段 {len(ts['blocks'])} 个月块：{len(examined)} 块记「{EXAMINED}」（被考 {ts['n_exams']} 次）"
        + ("" if not shared else f"；{'、'.join(shared)} 那块和 V 段共用一个月，记「{ts['block_states'][shared[0]]}」"))
    rs = reg.summary()
    say(f"  试验账本：N_T = {rs['N_T']}（含中止、放弃，不含校验拒绝）；n_prop = {rs['n_prop']}；送检 {rs['n_sent']} 个"
        f"（登记 n_fin = {rs['n_fin']}）；结论：{rs['verdict']}")
    nxt = ledger.register_v(MARKET, sp.v.start, sp.v.end, "演示任务-002")
    say(f"  下一条需求想再用这段 V → 照登记（B16，{nxt['verdict']}）：它是这段 V 的第 {nxt['n']} 个任务，"
        f"它的闸门报告和汇总表加注「{nxt['note']}」——只是加注，不判不可认证")
    out.update(v_status=vs, test_status=ts, registry_summary=rs, next_v=nxt)

    # ======== 第 9 步：审计记录行 ========
    say("\n【第 9 步】审计记录行（C1；C13 禁令句；C2：代码不许拿 L_label、m_ov 分支）")
    families = [T.FamilySchema(name, "合成市场里的候选特征，定义从略（演示）；合成市场的全部资产") for name in names]
    _, spec = T.build_schema_only_prompt(DEMO_DEMAND, families, (sp.t_1, sp.t_2))
    rec = R.make_call_record(TOP_MODEL, run_date, TOP_CUTOFF, sp.t_1.date(), sp.t_2.date(), prompt=spec)
    say("  顶层还没接：下面是假设的一次调用该记的一行（模型和截止月都是假设的）")
    say(f"  这次提示是 {spec.prompt_type}（只见 {len(families)} 个族的字段说明和测试窗日期，不带任何数值表），"
        f"禁令句按 C13 默认用 {spec.ban_variant}")
    for line in rec.render().splitlines():
        say("  " + line)
    say(f"  按 L_label 查表，日报里测试段成绩旁要标：「{rec.test_tag or '（不标）'}」")
    hits = R.static_branch_check([Path(R.__file__).resolve().parent, Path(__file__).resolve()])
    say("  " + R.format_hits(hits).replace("\n", "\n  "))
    out.update(audit=rec, branch_hits=hits, prompt_spec=spec)

    # ======== 第 10 步：影子与日报 ========
    say("\n【第 10 步】影子与日报（C4：冻结后每天用冻结权重在新行情上记分；C5：日报并排印测试段成绩、影子成绩、已跑天数）")
    new = shadow_market(seed, sp.t_2, max(SHADOW_DAYS) + 10)
    say(f"  冻结后的新行情（合成，另一个种子续造）：{_d(new.dates[0])} 起；影子脚本推理前先核对手上的权重是不是冻结的那份："
        f"{'是' if frozen.same_weights([float(direction)]) else '不是'}")
    pred_new, label_new = frozen_predictions(new, pick, direction), label_frame(new)
    noise_line = (f"按你的送检上限和验证期，纯噪音能碰到的成绩约 {G.e_max(N_FIN) * math.sqrt(252 / T_VAL):.2f}"
                  f"（D10，只展示；年化 Sharpe，E_max(n_fin)·√(252/T_val)，n_fin = {N_FIN}、T_val = {T_VAL}）")
    reports = []
    for n in SHADOW_DAYS:
        as_of = new.dates[n - 1]
        shadow = P.shadow_score(frozen, pred_new, labels_known_by(label_new, as_of, mk.label_horizon), as_of)
        reports.append(P.daily_report(frozen, shadow, as_of=as_of, exam=exam, L_label=rec.L_label,
                                      extra_lines=[noise_line]))
    say(f"  冻结后第 {SHADOW_DAYS[0]} 个交易日的日报（只给人看，不进汇总表；以下成绩全是合成数据）：")
    for line in reports[0]["lines"]:
        say("    " + line)
    say(f"  第 {SHADOW_DAYS[1]} 个交易日：{reports[1]['lines'][2]}")
    say(f"  已跑天数比交易日少 {mk.label_horizon} 天：最近两天的标签还没实现，影子只记已实现的。"
        f"可用性一直是「{P.AVAILABILITY}」——考试成绩和各种标签都只是贴在成品上的，不挡使用（B25）")
    exam_numbers = [f"{x:.4f}" for x in (exam.score.ic_mean, exam.score.ir) if x is not None]
    canary_ok = not any(x in text for x in exam_numbers)
    say(f"  金丝雀（A4，可选）：测试段成绩的数在日报里有、在汇总表里{'查不到' if canary_ok else '居然出现了'}（A7 / C6）")
    out.update(shadow_market=new, reports=reports, canary_ok=canary_ok)

    # ======== 第 11 步：收单演示 ========
    say("\n【第 11 步】收单演示：三条需求 → 拒 / 问 / 转译（D1–D8、D12；以拒结案的只有要保证、拼速度、没数据三种）")
    say("  正式流水线里收单在最前（第一节）；这里放最后，只因为这三条示例需求和主线需求无关")
    say("  用示例收单边界 EXAMPLE_BOUNDARY（4 个市场、示例代理、示例利率，不是登记值；默认边界 Boundary() 只登 A 股）")
    intake = []
    for k, raw in enumerate(REQUESTS, 1):
        a = I.assess(I.extract_card(raw), I.EXAMPLE_BOUNDARY)
        intake.append(a)
        say(f"  {k}. 「{raw}」")
        say(f"     处置：{a.disposition}；命中：{'、'.join(a.codes) or '无'}")
        for line in _sentences(a.message):
            say("       " + line)
        for h in a.header:
            say("       方案抬头（D9 / D10，只展示，不进汇总表）：" + h)
    out["intake"] = intake
    return out


# ---------------- 多世界附表 ----------------
def tally_worlds(seeds) -> dict:
    """多世界频率（B14 的演示核对）：每个世界从头造市场、粗筛、过闸（不走账本），数三类候选各过闸几次。
    用来说明演示种子挑得多狠：旧价候选「看着显著」只在一部分世界出现。"""
    t = {"n": 0, "plant_sent": 0, "plant_pass": 0, "stale_sent": 0, "stale_robust": 0, "stale_pass": 0,
         "noise_sent": 0, "noise_pass": 0, "good": []}
    for seed in seeds:
        world = build_world(seed)
        screen = coarse_screen(world.market.features, world.label, world.rows_s, N_FIN)
        selected = [r for r in screen if r.selected]
        t["n"] += 1
        plant_ok, stale_ok, noise_ok = False, False, True
        for r in selected:
            s_ret = candidate_returns(world, r.name, r.direction, world.rows_s)
            v_ret = candidate_returns(world, r.name, r.direction, world.rows_v)
            res = G.gate_decision(v_ret, len(selected), ALPHA_GATE, s_returns=s_ret, beta=BETA_POW,
                                  switch_to=SWITCH_TO, a1_alpha=A1_ALPHA)
            passed = res.verdict == G.PASS
            kind = world.market.kinds[r.name]
            if kind == PLANTED:
                t["plant_sent"] += 1
                t["plant_pass"] += passed
                plant_ok = passed
            elif kind == STALE:
                robust_pass = G.robust_t(v_ret) >= res.kappa
                t["stale_sent"] += 1
                t["stale_robust"] += robust_pass
                t["stale_pass"] += passed
                stale_ok = robust_pass and not passed and res.statistic != "robust_t"
            else:
                t["noise_sent"] += 1
                t["noise_pass"] += passed
                noise_ok = noise_ok and not passed
        if plant_ok and stale_ok and noise_ok:
            t["good"].append(seed)
    return t


def _pct(k: int, n: int) -> str:
    return f"{k}/{n}（{k / n:.1%}）" if n else f"{k}/0"


def print_tally(t: dict, seeds, say=print, demo_seed: int = DEMO_SEED) -> None:
    """印多世界附表（B14；单个候选的名义水平 α/N 同 B4 的 Bonferroni 那一项）。"""
    seeds = list(seeds)
    say(f"\n【附表】多世界频率：种子 {seeds[0]}–{seeds[-1]} 共 {t['n']} 个世界，每个都从头造市场、粗筛、过闸（不走账本）")
    say(f"  植入：送进 V {_pct(t['plant_sent'], t['n'])}；送进后过闸 {_pct(t['plant_pass'], t['plant_sent'])}")
    say(f"  旧价：送进 V {_pct(t['stale_sent'], t['n'])}；送进后若一律用稳健 t 会过闸 "
        f"{_pct(t['stale_robust'], t['stale_sent'])}，按 B14 换统计量后过闸 {_pct(t['stale_pass'], t['stale_sent'])}；"
        f"单个候选的名义水平 α/N = {ALPHA_GATE / N_FIN:.0%}")
    say(f"  噪声：送进 V 共 {t['noise_sent']} 个次；过闸 {_pct(t['noise_pass'], t['noise_sent'])}")
    good = t["good"]
    where = f"演示用的种子 {demo_seed} 在其中" if demo_seed in good else f"演示用的种子 {demo_seed} 不在这批里"
    say(f"  三件事同时看得到（植入过闸、旧价稳健 t 越过 κ 但被 B14 拦下、送进 V 的噪声全没过）的世界："
        f"{_pct(len(good), t['n'])}；{where}")
    say("  读法：旧价候选不是每个世界都「看着显著」；一律用稳健 t 时它过闸的频率比名义水平高，按 B14 换成 HAC t 后少了"
        "很多。按 B14 那一格样本太少，看不出有没有压回名义水平——--stale-mc 的直接蒙特卡洛给更准的数。")


def stale_null_check(n_paths: int, *, path_len: int = T_VAL, seed: int = 1000) -> dict:
    """旧价策略零分布的直接蒙特卡洛：另造几份很长的合成市场（不带噪声特征），把旧价策略收益切成 n_paths 段长
    path_len 天的路径，数稳健 t 与 HAC t 越过 κ(N_FIN, α_gate) 的比例（单侧）。
    只用来核对 B14「换 t 只修尺度」修到什么程度，不是流程的一部分。"""
    per_market = 150      # 每份市场切 150 段：工作日历再长就超出 pandas 的日期范围
    rob, hac = [], []
    k = 0
    while len(rob) < n_paths:
        mk = make_market(seed=seed + k, n_days=path_len * per_market + 10, n_noise=0)
        r = strategy_returns(mk.features[STALE], mk.label)
        r = r[np.isfinite(r)]
        for j in range(min(len(r) // path_len, n_paths - len(rob))):
            x = r[j * path_len:(j + 1) * path_len]
            rob.append(G.robust_t(x))
            hac.append(G.hac_t(x))
        k += 1
    kap = G.kappa(N_FIN, ALPHA_GATE)
    nominal = float(norm.sf(kap))
    p_rob, p_hac = float(np.mean(np.array(rob) >= kap)), float(np.mean(np.array(hac) >= kap))
    return {"n": len(rob), "path_len": path_len, "kappa": kap, "nominal": nominal, "p_star": ALPHA_GATE / N_FIN,
            "p_robust": p_rob, "p_hac": p_hac, "rho_robust": p_rob / nominal, "rho_hac": p_hac / nominal}


def print_stale_check(c: dict, say=print) -> None:
    """印旧价零分布的直接蒙特卡洛（B14 的核对）。"""
    say(f"\n【附表二】旧价策略零分布的直接蒙特卡洛（{c['n']} 条 {c['path_len']} 天的路径；均值真的是 0）")
    say(f"  越过 κ = {c['kappa']:.3f} 的比例：稳健 t {c['p_robust']:.1%}（是正态名义尾概率的 {c['rho_robust']:.1f} 倍），"
        f"HAC t {c['p_hac']:.1%}（{c['rho_hac']:.1f} 倍）；正态名义尾概率 {c['nominal']:.2%}，分界 α/N = {c['p_star']:.0%}")
    side = "仍高于" if c["p_hac"] > c["p_star"] else "不高于"
    say(f"  读法：换成 HAC t 把假过闸压小了，但{side}分界 α/N——换 t 只修尺度（B14）；这类候选就算过闸也只作参考（U-6）")


def main(argv=None) -> int:
    """命令行入口：python demo.py [--seed N] [--worlds K] [--stale-mc M]。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # 管道 / 重定向时 Windows 默认编码印不了中文
    ap = argparse.ArgumentParser(description="Stockburger 原型 · 合成数据集成演示（验收标准 v007）")
    ap.add_argument("--seed", type=int, default=DEMO_SEED, help=f"合成市场的种子（默认 {DEMO_SEED}，挑过的）")
    ap.add_argument("--worlds", type=int, default=200, help="多世界附表的世界数（0 = 不算附表）")
    ap.add_argument("--stale-mc", type=int, default=0, help="旧价策略零分布直接蒙特卡洛的路径数（0 = 不算；4500 约十几秒）")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    run_pipeline(args.seed)
    if args.worlds > 0:
        seeds = range(args.worlds)
        print_tally(tally_worlds(seeds), seeds, demo_seed=args.seed)
    if args.stale_mc > 0:
        print_stale_check(stale_null_check(args.stale_mc))
    print(f"\n（用时 {time.perf_counter() - t0:.1f} 秒）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
