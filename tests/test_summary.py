# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""汇总表的单测（v007.1 B1 / B5 / B11 / B16 / A4 / A7 / C6 / A8 / C8）。"""
import numpy as np
import pytest

from stockburger import gate as G
from stockburger import summary as S

CONSTS = S.Constants(n_t_tr=5, kappa=2.837, annual_threshold=2.837, sr_mde=3.68, rounds_left=7, alpha_gate=0.05)
KAPPA = CONSTS.kappa
ROWS = [S.SRow("c-03", 0.41, "已知"), S.SRow("c-11", 0.93), S.SRow("c-07", 0.66, "未匹配"),
        S.SRow("rule-跌5%买", 0.20, is_baseline=True)]


def build(**kw):
    args = dict(task_id="T1", round_no=3, consts=CONSTS, s_rows=ROWS)
    args.update(kw)
    return S.build_summary(**args)


def gate_verdicts(v_returns: dict, kappa: float) -> list:
    """测试里的闸门替身：V 段日收益 → t 值 → 二值（真闸门在 gate 模块）。"""
    out = []
    for cand, r in v_returns.items():
        t = r.mean() / r.std(ddof=1) * np.sqrt(len(r))
        out.append(S.VResult(cand, S.GATE_PASS if t >= kappa else S.GATE_FAIL))
    return out


def series(mean, sd=0.01, seed=0, n=252):
    noise = np.random.default_rng(seed).normal(0, 1, n)
    noise = (noise - noise.mean()) / noise.std(ddof=1)
    return mean + sd * noise          # 均值、标准差精确受控，t = mean / sd · √n


def test_default_L0_before_v_only_says_not_sent():
    table = build()
    text = S.render_summary(table)
    assert table["V区"]["结果"] == "尚未送检"
    assert "L = 0：S 段筛完一次送检，n_fin ≤ 5" in text
    assert S.S_LABEL in text and S.U6_NOTE in text
    assert "协议：完好" in text


def test_three_zones_content():
    table = build()
    assert set(table) == S.TOP_KEYS and set(table["S区"]) == S.S_KEYS
    assert table["常数区"]["N_T^tr"] == 5 and table["常数区"]["α_gate"] == 0.05
    s = table["S区"]
    assert [r["cand_id"] for r in s["候选"]] == ["c-11", "c-07", "c-03"]       # 按 S 段成绩排
    assert [r["cand_id"] for r in s["基线"]] == ["rule-跌5%买"]                 # A8：基线单列
    text = S.render_summary(table)
    assert "基线（人给的硬规则，A8）rule-跌5%买  0.200" in text


def test_constants_come_from_gate_constants():
    """常数区的前四项可以直接拿 gate.gate_constants 的结果填。"""
    c = G.gate_constants(5)
    consts = S.Constants(**{k: c[k] for k in ("n_t_tr", "kappa", "annual_threshold", "sr_mde")},
                         rounds_left=0, alpha_gate=0.05)
    text = S.render_summary(build(consts=consts))
    assert f"κ = {c['kappa']:.3f}" in text and f"SR_mde = {c['sr_mde']:.3f}" in text and "剩余轮数 = 0" in text


def test_L0_after_v_gives_verdicts_only():
    v = gate_verdicts({"c-11": series(0.003), "c-07": series(0.0001, seed=1)}, KAPPA)
    table = build(v_results=v)
    assert table["V区"]["结果"] == [{"cand_id": "c-11", "verdict": "过闸"}, {"cand_id": "c-07", "verdict": "未过闸"}]
    assert not any(isinstance(x, float) for x in S._all_values(table["V区"]))
    text = S.render_summary(table)
    assert "c-11：过闸" in text and "c-07：未过闸" in text
    broken = build(v_results=[("c-11", "不可认证")], protocol_issues=["登记被改"])   # 协议被破坏时印「不可认证」
    assert broken["V区"]["结果"][0]["verdict"] == "不可认证" and broken["V区"]["协议"] == "不可认证：登记被改"
    assert build(v_results=[("c-11", "不可认证")])["V区"]["协议"].startswith("不可认证")      # 协议行跟着变
    with pytest.raises(ValueError):
        build(v_results=[("c-11", "2.91")])


def test_perturbation_within_same_cell_gives_identical_text():
    """V 段收益换成同一格内另一组数，汇总表一字不差（V 区不漏任何 V 段数值）。"""
    world_a = {"c-11": series(0.0001, seed=1), "c-07": series(0.0006, seed=2)}
    world_b = {"c-11": series(-0.0005, sd=0.02, seed=3), "c-07": series(0.0012, sd=0.009, seed=4)}
    va, vb = gate_verdicts(world_a, KAPPA), gate_verdicts(world_b, KAPPA)
    assert va == vb                                       # 两个世界落在同一格（都未过闸）
    assert S.render_summary(build(v_results=va)) == S.render_summary(build(v_results=vb))
    vc = gate_verdicts({"c-11": series(0.003, seed=1), "c-07": series(0.0006, seed=2)}, KAPPA)
    assert S.render_summary(build(v_results=vc)) != S.render_summary(build(v_results=va))   # 跨格就该变


def test_v_numeric_field_is_rejected():
    with pytest.raises(ValueError, match="V 区只收"):
        build(v_results=[{"cand_id": "c-11", "verdict": "未过闸", "t_stat": 2.1}])


def test_no_test_segment_information():
    sentinel = "SENTINEL-TEST-9f2c"
    test_segment = {"sharpe": 9.99, "tag": sentinel}       # 测试段结果存在，但汇总表没有入口
    text = S.render_summary(build())
    assert sentinel not in text and "9.99" not in text and str(test_segment["sharpe"]) not in text
    with pytest.raises(ValueError):                         # 想从 S 区夹带也不行
        build(s_rows=[{"cand_id": "c-1", "s_score": 0.5, "test_sharpe": 9.99}])


def test_novelty_marks_only_two_readings():
    """C8：只有「已知」「未匹配」，不许写「新发现」。"""
    text = S.render_summary(build())
    assert "c-07  0.660  未匹配" in text and "c-03  0.410  已知" in text
    for bad in ("新发现", "已知（有年份）"):
        with pytest.raises(ValueError):
            build(s_rows=[S.SRow("c-1", 0.5, bad)])


def test_reuse_note_line_in_v_zone():
    """B16（v007.1）：这段 V 给过 n ≥ 2 个任务时 V 区多一行「复用」；只是加注，三值和协议都不变，白名单照过。"""
    table = build(v_results=[("c-11", S.GATE_FAIL)], reuse_note=G.reuse_note(2))
    assert table["V区"]["复用"] == "这段 V 给过 2 个任务：闸门结论更只能当参考" and table["V区"]["协议"] == "完好"
    assert S.check_whitelist(table) == []
    assert "  复用：这段 V 给过 2 个任务：闸门结论更只能当参考" in S.render_summary(table)
    plain = build(v_results=[("c-11", S.GATE_FAIL)], reuse_note=G.reuse_note(1))
    assert "复用" not in plain["V区"] and "复用" not in S.render_summary(plain)


def test_protocol_broken_shows_not_certifiable():
    table = build(protocol_issues=["登记被改"])
    assert table["V区"]["协议"] == "不可认证：登记被改"
    too_many = [(f"c-{i}", S.GATE_FAIL) for i in range(6)]
    table = build(v_results=too_many, n_fin=5)
    assert table["V区"]["协议"].startswith("不可认证") and "超过登记的 n_fin = 5" in table["V区"]["协议"]


def test_whitelist_catches_injected_fields():
    table = build()
    assert S.check_whitelist(table) == []
    table["V区"]["t_stat"] = 2.1
    table["S区"]["影子IC"] = 0.1
    table["常数区"]["进考场前审查"] = "通过"
    bad = S.check_whitelist(table)
    assert any("t_stat" in b for b in bad)
    assert any("连续值" in b for b in bad)
    assert any("影子IC" in b for b in bad) and any("审查" in b for b in bad)
    with pytest.raises(AssertionError):
        S.assert_whitelist(table)


def test_render_is_deterministic_and_order_free():
    assert S.render_summary(build()) == S.render_summary(build(s_rows=list(reversed(ROWS))))


def test_gate_warning_goes_to_constants_zone():
    text = S.render_summary(build(gate_warning=True))
    assert "警告：闸门无力认证" in text
