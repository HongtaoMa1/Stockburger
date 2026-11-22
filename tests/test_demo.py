# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""集成演示（demo.py）的测试：固定种子跑一遍主线，核演示该看到的三件事、切分与清洗、协议与两本账、汇总表白名单、
冻结与考试、影子与日报、审计记录行、收单三种处置；再换几个种子确认流程本身不挑世界。

三件事：噪声候选不过闸；植入特征过闸；「旧价」候选在稳健 t 下看着显著，但按 B14 换成 HAC t 后被拦下。
规格：验收标准 v007（条款大意见 README「条款编号对照」）。
"""
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # demo.py 在仓库根目录

import demo  # noqa: E402
from stockburger import gate as G  # noqa: E402
from stockburger import intake as I  # noqa: E402
from stockburger import product as P  # noqa: E402
from stockburger import records as R  # noqa: E402
from stockburger import segments as SG  # noqa: E402
from stockburger import summary as S  # noqa: E402
from stockburger.segments import BY_AI, EXAMINED, PROTOCOL_OK, SEG_S, SEG_TEST, SEG_V, UNCERTIFIABLE, USED  # noqa: E402
from stockburger.synth import NOISE, PLANTED, STALE  # noqa: E402

RUN_DATE = date(2026, 11, 14)


def _silent(*args, **kwargs):
    pass


@pytest.fixture(scope="module")
def run():
    return demo.run_pipeline(demo.DEMO_SEED, say=_silent, run_date=RUN_DATE)


# ---------------------------------------------------------------- 三件事
def test_planted_passes(run):
    assert PLANTED in run["selected"]
    res = run["results"][PLANTED]
    assert res.verdict == G.PASS and res.statistic == "robust_t" and res.t_value >= run["kappa"]
    assert res.guarantee == G.PASS_NOTE                                # B22：过闸者固定印「族错误率保证不适用 / 未核」


def test_stale_looks_significant_under_robust_t_but_b14_blocks_it(run):
    assert STALE in run["selected"]
    choice, res = run["choices"][STALE], run["results"][STALE]
    assert choice.a1_significant and choice.a1 > 0.3                  # S 段 a₁ 明显为正且显著
    assert res.statistic == demo.SWITCH_TO == "hac_t"                  # B14：换了统计量
    assert run["t_robust"][STALE] >= run["kappa"]                      # 一律用稳健 t 就会过闸
    assert res.t_value < run["kappa"] and res.verdict == G.FAIL       # 换统计量后没过
    assert not res.protocol_broken                                     # 未过闸 ≠ 不可认证（U-6）


def test_noise_candidates_fail(run):
    sent_noise = [n for n in run["selected"] if run["world"].market.kinds[n] == NOISE]
    assert sent_noise and len(run["selected"]) == demo.N_FIN
    assert all(run["results"][n].verdict == G.FAIL for n in sent_noise)


def test_gate_reports_fixed_sentences(run):
    for res in run["results"].values():                                # B10 认证范围、U-6 参考口径，固定印
        assert res.scope == G.SCOPE_NOTE and res.note == G.REFERENCE_NOTE and not res.violations
        assert G.SCOPE_NOTE in res.report()
    assert run["task_result"] == G.PASS and run["violations"] == []


def test_ai_pick_takes_highest_v_t_even_if_not_passed():
    """A2：没人时按登记的挑选规则代为确认——默认 V 段 t 最高的那个；闸门结果只作参考（U-6）。"""
    sel = [NS(name="a"), NS(name="b"), NS(name="c")]
    res = {"a": NS(t_value=1.0), "b": NS(t_value=2.5), "c": NS(t_value=None)}   # c：协议被破坏，没算 t
    assert demo.rank_by_v_t(sel, res) == ["b", "a", "c"]
    assert demo.ai_pick(sel, res)[0] == "b"
    with pytest.raises(RuntimeError):
        demo.ai_pick(sel[2:], res)


# ---------------------------------------------------------------- 切分与清洗（A1、A3）、登记（B3）
def test_split_and_purge_keep_every_sample(run):
    world = run["world"]
    sp = world.split
    assert (sp.s.n_days, sp.v.n_days, sp.test.n_days) == (992, demo.T_VAL, demo.T_TEST)
    assert int(world.rows_s.sum()) == sp.s.n_days and int(world.rows_v.sum()) == sp.v.n_days   # 照默认一个都不删
    assert run["purge"] == {"verdict": PROTOCOL_OK, "reasons": []}
    reg = run["registry"].registration
    assert reg["split_fingerprint"] == sp.fingerprint() and reg["d_pur"] == sp.d_pur
    assert (reg["n_fin"], reg["r_max"], reg["L"], reg["pick_rule"]) == (demo.N_FIN, demo.R_MAX, 0, demo.PICK_RULE)


# ---------------------------------------------------------------- 试验账本（B5、B7）
def test_registry_and_protocol(run):
    rs = run["registry_summary"]
    assert rs["verdict"] == PROTOCOL_OK and rs["reasons"] == []
    assert (rs["N_T"], rs["n_prop"], rs["n_sent"], rs["n_fin"]) == (22, 1, demo.N_FIN, demo.N_FIN)
    sp = run["world"].split
    rows = {r["cand_id"]: r for r in run["registry"].export_rows()}
    assert len(rows) == rs["N_T"] + rs["n_prop"]
    stale = rows[STALE]
    assert stale["scores"]["V_统计量"] == "hac_t" and stale["touched"] == (SEG_S, SEG_V)
    assert stale["s_returns"].index.max() < sp.v.start                                  # S 段收益只有 S 段的日子
    v = stale["v_returns"]
    assert len(v) == sp.v.n_days and v.index.min() == sp.v.start and v.index.max() == sp.v.end
    unsent = [r for c, r in rows.items() if c not in run["selected"] and r["status"] != "校验拒绝"]
    assert unsent and all(r["v_returns"] is None and r["touched"] == (SEG_S,) for r in unsent)
    assert all(SEG_TEST not in r["touched"] for r in rows.values())                     # 测试段只有考试脚本读（A2）


# ---------------------------------------------------------------- 日历块账本（B9、B16、B19、A2）
def test_block_ledger_and_single_exam(run):
    sp = run["world"].split
    vs, ts = run["v_status"], run["test_status"]
    acq, check = run["v_acq"], run["v_check"]
    assert (acq["start"], acq["end"]) == (sp.v.start, sp.v.end)                    # B19：V 就是紧挨测试段的那段，不挪
    assert (acq["n"], acq["note"]) == (1, "") and (check["n"], check["note"]) == (1, "")    # 第 1 个任务：不加注
    assert all(r.v_tasks == 1 and r.reuse_note == "" for r in run["results"].values())
    assert vs["state"] == USED and vs["used_by"] == [demo.TASK_ID] and vs["n_uses"] == 1
    assert ts["n_exams"] == 1
    assert all(st == EXAMINED for b, st in ts["block_states"].items() if b not in vs["blocks"])   # 和 V 共用的月除外
    start = run["exam_start"]
    assert start["allowed"] and start["k"] == 1 and start["confirmed_by"] == BY_AI and start["note"] == ""
    assert run["exam_resume"]["resumed"] and run["exam_resume"]["k"] == 1           # 断了重跑同一场：不算再考
    again = run["exam_again"]
    assert again["verdict"] == UNCERTIFIABLE and not again["allowed"]              # 同一需求再考 = 不可认证
    assert [r["event"] for r in run["ledger"].rows()].count(SG.EV_EXAM) == 1       # 重跑、被拒的再考都不另写行
    nxt = run["next_v"]                                                            # B16（v007.1）：一段 V 可以给多个任务用
    assert nxt["registered"] and nxt["verdict"] == PROTOCOL_OK and nxt["reasons"] == []
    assert (nxt["start"], nxt["end"]) == (sp.v.start, sp.v.end) and nxt["reused_from"] == [demo.TASK_ID]
    assert nxt["n"] == 2 and nxt["note"] == SG.REUSE_NOTE.format(n=2)               # 第 2 个任务起加注，不判不可认证


# ---------------------------------------------------------------- 汇总表（B1、A4、A7）
def test_summary_whitelist_and_v_zone(run):
    table = run["summary"]
    assert S.check_whitelist(table) == [] and table["schema"] == S.SCHEMA_VERSION
    c = table["常数区"]
    assert c["N_T^tr"] == demo.N_FIN and c["κ"] == pytest.approx(G.kappa(demo.N_FIN, demo.ALPHA_GATE))
    assert c["SR_mde"] == pytest.approx(G.sr_mde(demo.N_FIN, demo.T_VAL, demo.ALPHA_GATE, demo.BETA_POW))
    assert len(table["S区"]["候选"]) == 22
    v = table["V区"]["结果"]
    assert [r["cand_id"] for r in v] == run["selected"]
    assert {r["verdict"] for r in v} == {S.GATE_PASS, S.GATE_FAIL} and table["V区"]["协议"] == "完好"
    assert "复用" not in table["V区"]                                               # 第 1 个任务用这段 V：没有复用加注
    assert run["canary_ok"]


# ---------------------------------------------------------------- 冻结、考试、影子与日报（C3、B25、A2、C4、C5）
def test_freeze_exam_and_daily_report(run):
    frozen, exam, sp = run["frozen"], run["exam"], run["world"].split
    h = run["world"].market.label_horizon
    assert frozen.availability == P.AVAILABILITY and frozen.trained_on == "S" and frozen.tau_freeze > sp.t_2
    assert exam.fingerprint == frozen.fingerprint and exam.ai_confirmed and exam.nth_on_segment == 1
    assert exam.segment == (str(sp.t_1.date()), str(sp.t_2.date()))
    assert exam.score.n_days == sp.test.n_days - h and exam.score.ic_t > 0     # 末尾 H 天的标签还没实现
    first, later = run["reports"]
    for rep in run["reports"]:
        assert rep["availability"] == P.AVAILABILITY and rep["exam"]["fingerprint"] == frozen.fingerprint
    exam_line, shadow_line = first["lines"][1], first["lines"][2]
    assert f"［{P.AI_CONFIRMED}］" in exam_line and f"［{R.STRESS_TAG}］" in exam_line   # L_label = partial
    assert shadow_line.startswith(f"影子成绩：已跑 {demo.SHADOW_DAYS[0] - h} 个交易日") and P.T_NOTE in shadow_line
    assert first["shadow"]["first"] > str(sp.t_2.date())                       # 冻结前的行情一天也不进影子
    assert later["shadow"]["n_days"] == demo.SHADOW_DAYS[1] - h
    assert first["lines"][3].startswith("按你的送检上限和验证期，纯噪音能碰到的成绩约")   # D10 进日报


# ---------------------------------------------------------------- 审计记录行（C1、C2、C13）
def test_audit_record_and_static_branch_check(run):
    rec, spec = run["audit"], run["prompt_spec"]
    assert rec.L_label == "partial" and str(rec.t_c) == "2026-03-31"         # t_1 ≤ t_c < t_2
    assert 0 < rec.m_ov == pytest.approx(R.overlap_months(rec.t_c, rec.t_1, rec.t_2))
    assert (spec.prompt_type, spec.ban_variant) == ("R0", "W2")
    assert rec.ban_variant == "W2" and rec.ban_hash == spec.ban_hash
    assert run["branch_hits"] == []                                            # 本包与 demo.py 都不拿 L_label / m_ov 分支


# ---------------------------------------------------------------- 收单（D1–D12）
def test_intake_three_dispositions(run):
    assert [a.disposition for a in run["intake"]] == ["拒", "问", "转译"]
    assert "F1" in run["intake"][0].codes and "F6" in run["intake"][1].codes
    assert all(a.boundary_hash == I.EXAMPLE_BOUNDARY.hash() for a in run["intake"])


# ---------------------------------------------------------------- 别的世界：流程不挑种子
@pytest.mark.parametrize("seed", [0, 1, 11])
def test_pipeline_runs_on_other_seeds(seed):
    out = demo.run_pipeline(seed, say=_silent, run_date=RUN_DATE)
    assert out["registry_summary"]["verdict"] == PROTOCOL_OK
    assert len(out["selected"]) == demo.N_FIN and PLANTED in out["selected"]
    assert out["v_status"]["state"] == USED and out["exam_again"]["verdict"] == UNCERTIFIABLE
    assert S.check_whitelist(out["summary"]) == [] and out["canary_ok"]
    assert all(rep["availability"] == P.AVAILABILITY for rep in out["reports"]) and out["branch_hits"] == []


def test_tally_counts_demo_seed_as_good_world():
    t = demo.tally_worlds([demo.DEMO_SEED])
    assert t["n"] == 1 and t["good"] == [demo.DEMO_SEED]
    assert t["plant_pass"] == 1 and t["stale_robust"] == 1 and t["stale_pass"] == 0 and t["noise_pass"] == 0


def test_stale_null_check_small():
    c = demo.stale_null_check(300)
    assert c["n"] == 300 and c["path_len"] == demo.T_VAL
    assert c["p_robust"] > c["p_hac"]                  # HAC 压小了 a₁ 通道……
    assert c["p_robust"] > c["p_star"]                 # ……稳健 t 下假过闸远高于 α/N


def test_main_prints_utf8_without_crashing(capsys):
    assert demo.main(["--worlds", "0"]) == 0
    text = capsys.readouterr().out
    for mark in ("【第 0 步】", "【第 11 步】", "B14 拦下", "影子成绩：已跑", P.AI_CONFIRMED, "不可认证"):
        assert mark in text, mark
