# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""收单与拒单的单测（v007 第八节 D1–D8 / D12、第五节 D9 / D10、第九节「收单边界」）。

初始计划第 5 节的三个例子与回归用例都用示例边界 EXAMPLE_BOUNDARY 跑（示例市场表、代理表、r_cert = 4%、r_f = 4%）；
默认边界 Boundary() 照第九节只登实际接通的东西，另有几条单测。
"""
import json
import math

import pytest

from stockburger import intake as I
from stockburger.gate import e_max

B = I.EXAMPLE_BOUNDARY

EXAMPLE_1 = ("依托传统计量，做稀疏模型，看美股小盘股的月度动量；模型要比现有产品更稳健；"
             "一旦流动性危机发生，会有尾部风险；我相信小盘股的错误定价会均值回归。")
EXAMPLE_2 = "用辩证法给我做一个年化 50%、风险为 0 的策略"
EXAMPLE_3 = "追求高风险高收益、喜欢日内交易、关注伊朗局势"
INSULTS = ("蠢", "傻", "白痴", "笨蛋", "脑子", "智商")


def run(raw, boundary=B, **kw):
    return I.assess(I.extract_card(raw), boundary, **kw)


# ---- 初始计划第 5 节的三个例子 -----------------------------------------------
def test_example2_guarantee_with_high_target_is_rejected_F1():
    card = I.extract_card(EXAMPLE_2)
    assert card.zero_risk == 1 and card.r_target == pytest.approx(0.50) and card.m_pref == ["辩证法"]
    res = I.assess(card, B)
    assert res.disposition == "拒"
    assert [h.code for h in res.hits if h.action == I.REJECT] == ["F1"]
    assert "A9" in res.codes                                        # 辩证法映射不上：也记一笔「问」，但拒优先
    assert I.F1_REASON in res.message                               # D4 的固定理由
    assert "「风险为 0」" in res.message and "zero_risk = 1" in res.message and "r_cert = 4%" in res.message
    assert "如果字段读错了，直接回复更正" in res.message
    assert not any(w in res.message for w in INSULTS)
    assert res.header == ()                                         # 拒单不出方案抬头


def test_example3_is_translated_and_keeps_intraday():
    card = I.extract_card(EXAMPLE_3)
    assert (card.risk_lbl, card.obj_lbl, card.tau) == ("高", "激进", "日内")
    assert card.U == ["石油"] and card.source_of("U", "石油") == I.INFERRED   # 伊朗 → 石油 是顶层推断
    assert card.feats == ["伊朗局势"]
    res = I.assess(card, B)
    assert res.disposition == "转译"
    assert {"F3′", "F4′"} <= set(res.codes) and not ({"F1", "F2", "F4", "F6"} & set(res.codes))
    assert any("日内" in n and "自动执行" in n for n in res.notes) and "周期「日内」照留" in res.message
    assert any("能源股" in n and "版本 示例" in n for n in res.notes)          # 代理带版本与承载度说明
    assert "U = 石油（顶层推断：「伊朗」）" in res.echo
    assert "tau = 日内（用户原文：「日内」）" in res.echo


def test_example1_is_translated_with_no_substitution():
    card = I.extract_card(EXAMPLE_1)
    assert card.m_pref == ["传统计量", "稀疏模型"] and card.U == ["美股"]
    assert card.tail == "流动性危机" and card.v == "小盘股的错误定价会均值回归"
    res = I.assess(card, B)
    assert res.disposition == "转译" and res.codes == () and "零替换" in res.message
    assert any("纯噪音能碰到的成绩" in h for h in res.header)              # D10：只进方案抬头
    assert any("观点「小盘股的错误定价会均值回归」" in h and "尾部情景「流动性危机」" in h for h in res.header)   # D9


# ---- 回归用例（每次改规则全过）--------------------------------------------------
REGRESSION = [
    # (原文, 处置, 必须有的原因码, 不许有的原因码)
    (EXAMPLE_1, "转译", set(), {"F4"}),
    (EXAMPLE_1.replace("美股", "朝鲜股市"), "拒", {"F4"}, set()),
    (EXAMPLE_2, "拒", {"F1"}, set()),
    (EXAMPLE_3, "转译", {"F3′"}, {"F2"}),
    ("年化 8%、波动 12%", "转译", set(), {"F7", "F7′"}),
    ("年化 15%、最大回撤 20%", "转译", set(), {"F7", "F7′"}),
    ("年化 20%、波动 10%", "转译", set(), set()),                 # 目标隐含的 Sharpe 不触发问或拒（D9）
    ("低风险、年化 6%", "转译", {"F7"}, set()),
    ("我很保守，但想要翻倍", "问", {"F6"}, set()),
    ("零风险、年化 3%", "问", {"F1′"}, {"F1"}),                    # 目标不高于 r_cert：问
    ("做原油期权", "问", {"F3"}, set()),
    ("高频做市（亚秒级报撤单）", "拒", {"F2"}, set()),
    ("要托管机房才能做的A股策略", "拒", {"F2"}, set()),
    ("做日频短期反转", "转译", set(), {"F2", "F2′"}),
    ("我知道没有零风险的策略，做个稳一点的", "问", {"F9"}, {"F1"}),
    # D5 只认亚秒级 / 托管机房 / 逐笔订单簿报撤单；带否定的、当数据用的不算；「高频」「延迟套利」单独出现只问
    ("我不要高频，做A股日频", "转译", set(), {"F2", "F2″"}),
    ("做A股日频短期反转，不要高频", "转译", set(), {"F2", "F2″"}),
    ("不要托管机房，做A股日频", "转译", set(), {"F2"}),
    ("我们没有托管机房，做A股日频动量", "转译", set(), {"F2"}),
    ("做一个高频因子的日频A股策略", "转译", set(), {"F2", "F2″"}),
    ("用逐笔成交做A股日频选股", "转译", set(), {"F2", "F2″"}),
    ("用分钟级高频数据做A股日频选股", "转译", set(), {"F2"}),
    ("做高频交易", "问", {"F2″"}, {"F2"}),
    ("做A股延迟套利", "问", {"F2″"}, {"F2"}),
    # D7 只在卡上的市场一个都没接通时拒；带否定的市场不进卡
    ("不做美股，只做A股日频动量", "转译", set(), {"F4"}),
    # D4：带否定的保证词不置 1，交给 D3 问
    ("不追求零风险，年化 20%，做A股", "问", {"F9"}, {"F1"}),
    ("零风险是不可能的，年化 20%，做A股", "问", {"F9"}, {"F1"}),
]


@pytest.mark.parametrize("raw, disposition, must, must_not", REGRESSION, ids=[r[0][:12] for r in REGRESSION])
def test_regression_set(raw, disposition, must, must_not):
    res = run(raw)
    assert res.disposition == disposition, (raw, res.codes)
    assert must <= set(res.codes), (raw, res.codes)
    assert not (must_not & set(res.codes)), (raw, res.codes)


def test_only_F1_F2_F4_can_reject():
    for raw, *_ in REGRESSION:
        for boundary in (B, I.Boundary()):
            res = run(raw, boundary)
            assert {h.code for h in res.hits if h.action == I.REJECT} <= {"F1", "F2", "F4"}
            assert (res.disposition == "拒") == any(h.action == I.REJECT for h in res.hits)


def test_rule_column_is_the_v007_clause():
    rules = {h.rule for raw, *_ in REGRESSION for h in run(raw).hits}
    assert rules <= {"D3", "D4", "D5", "D6", "D7", "D8", "A9"}


# ---- 单条规则 -----------------------------------------------------------------
def test_floor_guarantee_is_rejected_too():
    res = run("保本、年化 10%")
    assert res.disposition == "拒" and "F1" in res.codes and "「保本」（floor = 1）" in res.message


def test_derivatives_are_asked_never_rejected():
    res = run("做原油期权")
    assert res.disposition == "问" and not any(h.action == I.REJECT for h in res.hits)
    assert {"F3", "F3″"} <= set(res.codes)


def test_facility_phrase_mismatch_asks_instead_of_rejecting():
    res = I.assess(I.RequirementCard(raw="高频做市"), B)          # 原文说高频，卡上的机制却是空的
    assert res.disposition == "问" and "F9′" in res.codes and "F2" not in res.codes


def test_facility_word_used_as_data_only_noted():
    card = I.extract_card("用逐笔数据做A股选股")                  # 「逐笔数据」是特征，不是速度设施
    assert card.mech == [] and card.U == ["A股"]
    res = I.assess(card, B)
    assert res.disposition == "转译" and any(h.code == "F9′" and h.action == I.NOTE for h in res.hits)


def test_excluded_facilities_are_the_three_v007_kinds():
    """D5：拒只认亚秒级反应、托管机房、逐笔订单簿报撤单；「高频」「逐笔」「延迟套利」单独出现只问（F2″），不拒。"""
    assert set(I.Boundary().excluded_facilities) == {"亚秒级", "毫秒级", "微秒级", "托管机房", "逐笔报撤单"}
    for raw in ("做高频交易", "做A股延迟套利", "高频做市"):
        for boundary in (B, I.Boundary()):
            res = run(raw, boundary)
            assert res.disposition == "问" and "F2″" in res.codes and "F2" not in res.codes, (raw, res.codes)
    assert "F2′" in run("高频做市").codes                              # 做市照样给慢版本的转译注明


def test_negated_facility_and_market_words_stay_off_the_card():
    card = I.extract_card("我不要高频，也不要托管机房，做A股日频")
    assert card.mech == [] and card.U == ["A股"]
    notes = [h for h in I.assess(card, I.Boundary()).hits if h.code == "F9′"]
    assert notes and all(h.action == I.NOTE and "带否定" in h.message for h in notes)
    assert I.extract_card("不做美股，只做A股日频动量").U == ["A股"]
    assert I.extract_card("不做日内，做美股").U == ["美股"]               # 否定只在同一分句里找


def test_partly_connected_markets_translate_not_reject():
    """D7：卡上有一个市场接通就不拒，没接通的在回显里注明（F4″）；全都没接通才拒（F4）。"""
    b = I.Boundary()                                                    # 默认边界只登了 A 股
    res = run("参考美股的动量研究，做A股日频", b)
    assert res.disposition == "转译" and "F4″" in res.codes and "F4" not in res.codes
    assert any("只做已接通的「A股」" in n for n in res.notes)
    assert run("做美股和港股", b).disposition == "拒" and run("做美股和港股", b).codes.count("F4") == 2


def test_disputed_field_reading_asks_instead_of_rejecting():
    """D1–D3：用户不认字段读法就不拒——那条拒降成问（F9″），附原来的理由，请用户改写。"""
    res = run(EXAMPLE_2, disputed={"zero_risk"})
    assert res.disposition == "问" and "F9″" in res.codes and "F1" not in res.codes
    assert not any(h.action == I.REJECT for h in res.hits) and I.F1_REASON in res.message
    assert run("高频做市（亚秒级报撤单）", disputed={"mech"}).disposition == "问"
    assert run(EXAMPLE_2, disputed={"mech"}).disposition == "拒"         # 不认的是别的字段：照拒


def test_race_first_wave_is_translated_to_slow_version():
    res = run("做美股抢第一波新闻的策略")
    assert res.disposition == "转译" and "F2′" in res.codes and any("事件后的延续" in n for n in res.notes)


def test_all_named_features_deleted_asks_F8():
    res = run("用五角大楼披萨指数做美股日频策略")
    assert res.disposition == "问" and {"F4′", "F8"} <= set(res.codes)


def test_unknown_market_reject_names_the_field():
    res = run(EXAMPLE_1.replace("美股", "朝鲜股市"))
    assert "「朝鲜股市」" in res.message and "数据登记表" in res.message


def test_same_card_same_result_and_boundary_hash():
    card = I.extract_card(EXAMPLE_3)
    assert I.assess(card, B) == I.assess(card, I.EXAMPLE_BOUNDARY)
    assert I.Boundary().hash() == I.Boundary().hash() != B.hash()
    assert I.Boundary(r_tbill=0.05).hash() != I.Boundary().hash()


def test_waiver_lifts_asks_but_not_rejects():
    res = run("我很保守，但想要翻倍", waived={"F6"})
    assert res.disposition == "转译" and "F6" not in res.codes and any(h.code == "F6" and h.waived for h in res.hits)
    assert run(EXAMPLE_2, waived={"F1", "A9"}).disposition == "拒"


def test_card_fields_with_sources():
    assert I.FIELD_NAMES == ("raw", "r_target", "zero_risk", "floor", "sigma_max", "mdd_max", "risk_lbl", "obj_lbl",
                             "tau", "U", "inst", "mech", "feats", "m_pref", "v", "tail")
    card = I.extract_card(EXAMPLE_2)
    assert card.sources["r_target"] == (I.USER, "年化 50%")
    with pytest.raises(ValueError):
        card.put("budget", "训几天")                                 # v007 不要的字段已删


def test_header_noise_level_by_n_fin_and_target_sharpe():
    res = run("年化 8%、波动 12%")
    noise = e_max(5) / math.sqrt(252 / 252)                         # D10：E_max 按 n_fin 算
    assert any(f"纯噪音能碰到的成绩约 {noise:.2f}" in h for h in res.header)
    assert any("Sharpe ≈ 0.33" in h for h in res.header)            # (8% − 4%) / 12%
    p = 1 - I.norm.cdf(0.04 / 0.12) ** 5
    assert any(f"越过它的概率 ≈ {p:.2f}" in h for h in res.header)
    assert any("默认用年化 Sharpe" in h for h in res.header)         # D9：需求没说口径
    assert run("做原油期权").header == ()


# ---- 默认边界：照第九节只登实际接通的东西 ---------------------------------------
def test_default_boundary_registers_only_what_is_connected():
    b = I.Boundary()
    assert b.markets == ("A股",) and b.feature_proxies == () and b.exposure_proxies == () and b.r_cert is None
    assert run(EXAMPLE_1, b).disposition == "拒" and "F4" in run(EXAMPLE_1, b).codes    # 美股没接价格源
    assert run("用成交量做A股日频选股", b).disposition == "转译"
    res = run("用换手率做A股日频选股", b)                                                   # 数据里没有换手率、也没登记代理
    assert res.disposition == "问" and {"F4′", "F8"} <= set(res.codes)
    note = next(h for h in run(EXAMPLE_3, b).hits if h.code == "F3′")
    assert "代理还没登记" in note.message


def test_guarantee_without_registered_r_cert_asks_not_rejects():
    """第九节：r_cert 不用示例数；两样观测值没登记齐就比不出高低，只问不拒。"""
    res = run("零风险、年化 50%、做A股", I.Boundary())
    assert res.disposition == "问" and "F1′" in res.codes and "F1" not in res.codes
    assert "还没有 r_cert 的观测值" in res.message
    assert I.Boundary(r_tbill=0.014).r_cert is None                                        # 只登记一样也不算
    assert I.Boundary(r_tbill=0.014, r_deposit=0.015).r_cert == 0.015
    assert run("零风险、年化 50%、做A股", I.Boundary(r_tbill=0.014, r_deposit=0.015)).disposition == "拒"


# ---- 拒单日志 -----------------------------------------------------------------
def test_log_is_append_only_with_v007_fields(tmp_path):
    card = I.extract_card(EXAMPLE_2)
    res = I.assess(card, B)
    entry = I.make_log_entry("R-001", card, res, boundary=B)
    assert set(entry) == {"time", "req_id", "raw", "card", "rules_version", "boundary_hash", "hits", "disposition",
                          "message", "terminal"}
    assert entry["terminal"] == "拒" and entry["boundary_hash"] == B.hash() and "v007" in entry["rules_version"]
    assert entry["card"]["sources"]["r_target"] == [I.USER, "年化 50%"]
    path = tmp_path / "intake_log.jsonl"
    I.append_log(path, entry)
    ask = run("做原油期权")
    I.append_log(path, I.make_log_entry("R-002", I.extract_card("做原油期权"), ask, boundary=B))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["req_id"] == "R-001"
    assert json.loads(lines[1])["terminal"] == "等待回复"
    stuck = I.make_log_entry("R-003", I.extract_card("做原油期权"), ask, clarify_rounds=B.k_clarify, boundary=B)
    assert stuck["terminal"] == "挂起"                                # 反问轮数到上限：挂起，不是拒（D8）
    waived = run("我很保守，但想要翻倍", waived={"F6"})
    assert I.make_log_entry("R-004", I.extract_card("我很保守，但想要翻倍"), waived)["terminal"] == "用户覆盖"
    with pytest.raises(ValueError):
        I.append_log(path, {**entry, "terminal": "随便"})
