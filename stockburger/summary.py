# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""顶层每轮读的汇总表（B1）。

顶层每轮只读这张表决定下一轮，由脚本拼，分三区：
- 常数区：N_T^tr、κ、年化门槛、SR_mde、剩余轮数（按计划的 R_max 算）、α_gate；
- S 区：S 段成绩（DSR 形式，旁边标「展示用」）与人给的硬规则基线（单列，A8），可带新颖性标记（C8）；
- V 区：默认 L = 0、送检数 n_fin ≤ 5（B11），每个送检候选只给「过闸 / 未过闸」，协议被破坏时印「不可认证」
  （B5），不放任何 V 段数值——顶层看一个 V 段数值再决定下一轮，等于间接比较了无数个候选，门槛就算不出来。
  这段 V 给过 n ≥ 2 个任务时多一行「复用」加注（B16，v007.1：闸门结论更只能当参考），不判不可认证。
测试段成绩、影子成绩、审计账（含泄漏标记）、进考场前审查、[GUESS] 标注、噪音可达成绩一律不进本表（A7 / C6）；
字段白名单只是简单断言（A4 可选，U-5）。

规格：验收标准 v007（v007.1）的 B1、B5、B11、B16、A4、A7 / C6、A8、C8。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

SCHEMA_VERSION = "summary-v007"

S_LABEL = "展示用：已用于选择，不再有紧缩含义"                                   # B1 固定标签
U6_NOTE = "「过闸 / 未过闸」只作参考，不拦候选（U-6）；只有「不可认证」（协议被破坏）才算真的不过"

GATE_PASS, GATE_FAIL, NOT_CERTIFIABLE = "过闸", "未过闸", "不可认证"
V_VERDICTS = (GATE_PASS, GATE_FAIL, NOT_CERTIFIABLE)
NOVELTY_MARKS = ("已知", "未匹配")          # C8：跟已知库比相似度的标记，不许写「新发现」

# 白名单（简单断言用）
TOP_KEYS = {"schema", "task_id", "round", "常数区", "S区", "V区"}
CONST_KEYS = {"N_T^tr", "κ", "年化门槛", "SR_mde", "剩余轮数", "α_gate", "警告"}
S_KEYS = {"说明", "基线", "候选"}
S_ROW_KEYS = {"cand_id", "s_score", "novelty"}
V_KEYS = {"模式", "说明", "结果", "协议", "复用"}
V_ROW_KEYS = {"cand_id", "verdict"}
# 键名里出现这些词就算越界：测试段、影子、审计账、泄漏标记、进考场前审查、[GUESS] 标注、噪音可达成绩不进本表
BANNED_KEY_TOKENS = ("test", "测试", "shadow", "影子", "audit", "审计", "泄漏", "m_ov", "l_label",
                     "审查", "guess", "噪音")


@dataclass(frozen=True)
class Constants:
    """常数区（B1 / B3）：登记后由脚本算好传进来（gate.gate_constants 给前四个），汇总表只照抄。"""
    n_t_tr: int               # N_T^tr：闸门实际做的检验数（L = 0 时 = n_fin）
    kappa: float              # κ(N_T^tr, α_gate)
    annual_threshold: float   # 年化门槛
    sr_mde: float             # SR_mde
    rounds_left: int          # 剩余轮数（按计划的 R_max 算）
    alpha_gate: float         # α_gate


@dataclass(frozen=True)
class SRow:
    """S 区一行：候选与它的 S 段成绩（DSR 形式，N 取 N_T；B1）。"""
    cand_id: str
    s_score: float
    novelty: str | None = None    # C8：「已知」/「未匹配」，可选
    is_baseline: bool = False     # A8：人给的硬规则当基线，S 区单列


@dataclass(frozen=True)
class VResult:
    """V 区一行：只收候选 ID 与闸门三值之一。"""
    cand_id: str
    verdict: str                  # 过闸 / 未过闸 / 不可认证


def _as_srow(x) -> SRow:
    if isinstance(x, SRow):
        return x
    extra = set(x) - S_ROW_KEYS - {"is_baseline"}
    if extra:
        raise ValueError(f"S 区只收白名单字段，多出 {sorted(extra)}（A4 / B1）")
    return SRow(**x)


def _as_vresult(x) -> VResult:
    if isinstance(x, VResult):
        return x
    if isinstance(x, tuple):
        return VResult(*x)
    extra = set(x) - V_ROW_KEYS
    if extra:
        raise ValueError(f"V 区只收候选 ID 与闸门结果，多出 {sorted(extra)}（B1）")
    return VResult(**x)


def build_summary(task_id: str, round_no: int, consts: Constants, s_rows: Iterable, *, n_fin: int = 5,
                  v_results: Iterable | None = None, protocol_issues: Iterable[str] = (),
                  gate_warning: bool = False, reuse_note: str = "") -> dict:
    """生成顶层每轮读的汇总表（B1）。

    - 常数区照抄 consts；gate_warning=True 时挂「闸门无力认证」警告（B5，只提示）。
    - S 区：候选按 S 段成绩从高到低排（可带新颖性标记），人给的基线单列（A8），说明里标「展示用」。
    - V 区（L = 0，B11）：送检前印「尚未送检」；送检后每个候选只给三值之一，不给任何数值。
    - protocol_issues 非空、或送检数超过 n_fin → V 区「协议」印「不可认证」（B5）。
    - reuse_note 非空（这段 V 给过 n ≥ 2 个任务，传 gate.reuse_note(n) 或 BlockLedger.check_v 的 note）→ V 区多一行
      「复用」（B16，v007.1）：只是加注，不改三值、不判不可认证。
    本函数不收任何测试段输入；结尾做一次白名单断言（U-5：只是简单断言）。
    """
    rows = [_as_srow(r) for r in s_rows]
    for r in rows:
        if r.novelty is not None and r.novelty not in NOVELTY_MARKS:
            raise ValueError(f"新颖性标记只有 {NOVELTY_MARKS} 两种，收到 {r.novelty!r}（C8：不许写「新发现」）")
    baseline = [r for r in rows if r.is_baseline]
    cands = sorted((r for r in rows if not r.is_baseline), key=lambda r: (-r.s_score, r.cand_id))

    const_zone = {
        "N_T^tr": consts.n_t_tr,
        "κ": consts.kappa,
        "年化门槛": consts.annual_threshold,
        "SR_mde": consts.sr_mde,
        "剩余轮数": consts.rounds_left,
        "α_gate": consts.alpha_gate,
    }
    if gate_warning:
        const_zone["警告"] = "闸门无力认证（SR_mde 高于需求目标或历史放不下三段；只提示，任务照跑）"

    s_zone = {
        "说明": f"S 段成绩（DSR 形式，N 取 N_T）——{S_LABEL}",
        "基线": [{"cand_id": r.cand_id, "s_score": r.s_score} for r in baseline],
        "候选": [_s_row_dict(r) for r in cands],
    }

    issues = list(protocol_issues)
    v_zone = {"模式": f"L = 0：S 段筛完一次送检，n_fin ≤ {n_fin}", "说明": U6_NOTE}
    if v_results is None:
        v_zone["结果"] = "尚未送检"
    else:
        results = [_as_vresult(v) for v in v_results]
        for v in results:
            if v.verdict not in V_VERDICTS:
                raise ValueError(f"V 区只给 {V_VERDICTS} 之一，收到 {v.verdict!r}（B1：不放 V 段数值）")
        if len(results) > n_fin:
            issues.append(f"送检数 {len(results)} 超过登记的 n_fin = {n_fin}")
        if not issues and any(v.verdict == NOT_CERTIFIABLE for v in results):
            issues.append("闸门判了「不可认证」（协议被破坏）")
        v_zone["结果"] = [{"cand_id": v.cand_id, "verdict": v.verdict} for v in results]
    v_zone["协议"] = f"{NOT_CERTIFIABLE}：" + "；".join(issues) if issues else "完好"
    if reuse_note:
        v_zone["复用"] = str(reuse_note)

    table = {
        "schema": SCHEMA_VERSION,
        "task_id": str(task_id),
        "round": int(round_no),
        "常数区": const_zone,
        "S区": s_zone,
        "V区": v_zone,
    }
    assert_whitelist(table)
    return table


def _s_row_dict(r: SRow) -> dict:
    d = {"cand_id": r.cand_id, "s_score": r.s_score}
    if r.novelty is not None:
        d["novelty"] = r.novelty
    return d


def check_whitelist(table: dict) -> list[str]:
    """字段白名单的简单断言（B1，A4 可选；U-5：只是简单断言）。返回违规项列表，空 = 通过。

    查：各区键名在白名单内；键名不带测试段 / 影子 / 审计 / 泄漏等字样；V 区每行只有三值之一，
    V 区里不出现任何浮点数。"""
    bad = [f"顶层多出键 {k!r}" for k in table if k not in TOP_KEYS]
    allowed = {"常数区": CONST_KEYS, "S区": S_KEYS, "V区": V_KEYS}
    for zone, keys in allowed.items():
        bad += [f"{zone} 多出键 {k!r}" for k in table.get(zone, {}) if k not in keys]
    for k in _all_keys(table):
        if any(tok in str(k).lower() for tok in BANNED_KEY_TOKENS):
            bad.append(f"键名 {k!r} 带禁入字样（测试段、影子、审计账、泄漏标记等不进汇总表）")
    s_zone = table.get("S区", {})
    for row in list(s_zone.get("候选", [])) + list(s_zone.get("基线", [])):
        bad += [f"S 区行多出字段 {k!r}" for k in row if k not in S_ROW_KEYS]
    v_zone = table.get("V区", {})
    results = v_zone.get("结果")
    if isinstance(results, list):
        for row in results:
            bad += [f"V 区行多出字段 {k!r}" for k in row if k not in V_ROW_KEYS]
            if row.get("verdict") not in V_VERDICTS:
                bad.append(f"V 区结果不是三值之一：{row.get('verdict')!r}")
    if any(isinstance(x, float) for x in _all_values(v_zone)):
        bad.append("V 区出现连续值（不放任何 V 段数值）")
    return bad


def assert_whitelist(table: dict) -> None:
    """白名单断言：有违规就抛 AssertionError（显式抛，不受 python -O 影响）。"""
    bad = check_whitelist(table)
    if bad:
        raise AssertionError("汇总表白名单不过：" + "；".join(bad))


def _all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_keys(v)


def _all_values(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _all_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_values(v)
    else:
        yield obj


def _num(x) -> str:
    """确定性的数字格式：整数原样，浮点保留 3 位小数。"""
    return str(x) if isinstance(x, int) else f"{x:.3f}"


def render_summary(table: dict) -> str:
    """把汇总表写成顶层读的文本，逐字节确定：同一张表永远同一串字符，S 区行序与输入顺序无关。"""
    c, s, v = table["常数区"], table["S区"], table["V区"]
    out = [f"汇总表 · 任务 {table['task_id']} · 第 {table['round']} 轮 · {table['schema']}",
           "【常数区】（碰 V 前登记，与数据无关）"]
    out += [f"  {k} = {_num(c[k])}" for k in ("N_T^tr", "κ", "年化门槛", "SR_mde", "剩余轮数", "α_gate")]
    if "警告" in c:
        out.append(f"  警告：{c['警告']}")
    out.append(f"【S 区】{s['说明']}")
    for r in s["基线"]:
        out.append(f"  基线（人给的硬规则，A8）{r['cand_id']}  {_num(r['s_score'])}")
    for r in s["候选"]:
        tail = f"  {r['novelty']}" if "novelty" in r else ""
        out.append(f"  {r['cand_id']}  {_num(r['s_score'])}{tail}")
    out.append(f"【V 区】{v['模式']}；{v['说明']}")
    if isinstance(v["结果"], str):
        out.append(f"  {v['结果']}")
    else:
        out += [f"  {r['cand_id']}：{r['verdict']}" for r in v["结果"]]
    out.append(f"  协议：{v['协议']}")
    if "复用" in v:
        out.append(f"  复用：{v['复用']}")
    return "\n".join(out) + "\n"
