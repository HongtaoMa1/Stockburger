# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""段切分、清洗与日历块账本的单测（v007.1 A1 / A2 / A3 / B3 / B5 / B9 / B16 / B19）。"""
import json

import pandas as pd
import pytest

from stockburger import gate as G
from stockburger.segments import (
    BY_AI, BY_HUMAN, EXAMINED, GAP, NO_CLEAN_V, NOT_PASSED, OUTSIDE, PASSED, PROTOCOL_OK, RETEST_NOTE, REUSE_NOTE,
    SEG_S, SEG_TEST, SEG_V, UNCERTIFIABLE, UNUSED, USED, BlockLedger, jsonl_append, jsonl_rows, split_segments,
)

DATES = pd.bdate_range("2015-01-01", periods=2000)
MKT, V0, V1 = "CN", "2023-01-01", "2023-12-31"
T0, T1 = "2024-01-01", "2024-12-31"


def _seg_dates(seg):
    return DATES[(DATES >= seg.start) & (DATES <= seg.end)]


# ---------------- 切分（A1）----------------

def test_split_three_segments_in_order_with_gaps():
    sp = split_segments(DATES, t_val=252, t_test=252, label_horizon=2)
    pos = DATES.get_loc
    assert sp.t_2 == DATES[-1]                                   # t_2 = 数据末日
    assert sp.test.n_days == 252 and sp.v.n_days == 252
    assert sp.s.start == DATES[0] and sp.s.n_days == 2000 - 252 - 2 - 252 - 2
    assert sp.s.end < sp.v.start < sp.v.end < sp.test.start      # S < V < 测试
    assert pos(sp.v.start) - pos(sp.s.end) - 1 == sp.d_pur == 2  # 段间空档 = d_pur = 标签期
    assert pos(sp.test.start) - pos(sp.v.end) - 1 == 2
    assert sp.ranges()[SEG_TEST] == (sp.test.start, DATES[-1])
    assert sp.warnings == ()


def test_d_pur_defaults_to_label_horizon_and_cannot_be_smaller():
    assert split_segments(DATES, label_horizon=5).d_pur == 5
    sp = split_segments(DATES, label_horizon=2, d_pur=10)
    assert DATES.get_loc(sp.test.start) - DATES.get_loc(sp.v.end) - 1 == 10
    with pytest.raises(ValueError, match="d_pur"):
        split_segments(DATES, label_horizon=5, d_pur=3)


def test_unsorted_duplicated_dates_give_same_split():
    messy = list(DATES[::-1]) + list(DATES[:50])
    a, b = split_segments(DATES), split_segments(messy)
    assert a.ranges() == b.ranges() and a.fingerprint() == b.fingerprint()


def test_history_too_short_only_warns_but_impossible_split_raises():
    short = DATES[: 252 + 2 + 252 + 2]          # 正好放下 V + 测试 + 两个空档，S 段为空
    sp = split_segments(short)
    assert sp.s is None and any("闸门无力认证" in w for w in sp.warnings)
    sp2 = split_segments(DATES[:600], min_s_days=200)   # S 段 92 天 < 200：只提示
    assert sp2.s.n_days == 92 and any("闸门无力认证" in w for w in sp2.warnings)
    with pytest.raises(ValueError, match="放不下"):
        split_segments(DATES[:400])


def test_assign_labels_every_date():
    sp = split_segments(DATES)
    lab = sp.assign(DATES)
    assert (lab == SEG_S).sum() == sp.s.n_days
    assert (lab == SEG_V).sum() == 252 and (lab == SEG_TEST).sum() == 252
    assert (lab == GAP).sum() == 2 * sp.d_pur
    outside = sp.assign(["2014-12-01", "2030-01-02"])                  # 数据范围之外
    assert list(outside) == [OUTSIDE, OUTSIDE]


# ---------------- 清洗（A3）----------------

def test_default_gap_purges_nothing():
    """A3：段间空档 = 标签期时，S、V 段样本的标签窗末端都早于下一段首日，一个样本都不用删。"""
    sp = split_segments(DATES, label_horizon=2)
    s_dates, v_dates = _seg_dates(sp.s), _seg_dates(sp.v)
    assert sp.label_end([s_dates[0]])[0] == DATES[2]                     # 标签窗末端 = 之后第 2 个交易日
    assert sp.purge_boundary(SEG_S) == sp.v.start and sp.purge_boundary(SEG_V) == sp.t_1
    assert sp.purge_keep(SEG_S, s_dates).all() and sp.purge_keep(SEG_V, v_dates).all()
    assert sp.label_end(s_dates[-1:])[0] < sp.v.start and sp.label_end(v_dates[-1:])[0] < sp.t_1
    assert sp.purge_check(sp.label_end(s_dates), sp.label_end(v_dates)) == {"verdict": PROTOCOL_OK, "reasons": []}
    assert not sp.purge_keep(SEG_S, v_dates).any()                      # 不在本段的样本不留


def test_label_window_longer_than_gap_is_purged_and_checked():
    """A3：标签窗伸进下一段的样本一律删；不删就过不了清洗检查（B5：不可认证）。"""
    sp = split_segments(DATES, label_horizon=2)                         # 空档 2 天，标签窗按 5 天算
    s_dates, v_dates = _seg_dates(sp.s), _seg_dates(sp.v)
    raw = sp.purge_check(sp.label_end(s_dates, horizon=5), sp.label_end(v_dates, horizon=5))
    assert raw["verdict"] == UNCERTIFIABLE and len(raw["reasons"]) == 2
    ks, kv = sp.purge_keep(SEG_S, s_dates, horizon=5), sp.purge_keep(SEG_V, v_dates, horizon=5)
    assert (~ks).sum() == 3 and (~kv).sum() == 3 and not ks[-3:].any()  # 5 − 2 = 3 天伸进下一段
    ok = sp.purge_check(sp.label_end(s_dates[ks], horizon=5), sp.label_end(v_dates[kv], horizon=5))
    assert ok["verdict"] == PROTOCOL_OK
    assert sp.purge_check(s_label_ends=[pd.NaT])["verdict"] == UNCERTIFIABLE    # 标签终点缺失按不满足算


def test_keep_before_purges_train_rows_before_a_validation_fold():
    """A3：S 段内部切验证折也照此清洗——训练样本的标签窗末端要早于验证折首日。"""
    sp = split_segments(DATES)
    s_dates = _seg_dates(sp.s)
    keep = sp.keep_before(s_dates[:1000], s_dates[1000])
    assert keep[:-2].all() and not keep[-2:].any()                      # 标签期 2：最后 2 天的标签伸进验证折
    assert sp.keep_before(DATES[-1:], "2030-01-02").tolist() == [False]  # 数据末日的标签还没实现


# ---------------- 指纹（B3）----------------

def test_fingerprint_detects_changed_split():
    assert split_segments(DATES).fingerprint() != split_segments(DATES, t_val=200).fingerprint()


def test_fingerprint_algorithm_is_frozen():
    """已登记的指纹是按这套算法算的，算法一改就对不上——拿固定日历钉死。"""
    assert split_segments(DATES).fingerprint() == "efc1cbf78be7bc52f41e93c697f5509c84db517b6cabec17ddda66ef4895b3b8"


# ---------------- 日历块账本：V 段（B9 / B16；v007.1：一段 V 可以给多个任务用）----------------

def _run_v(led, tid, start=V0, end=V1, result=NOT_PASSED, market=MKT):
    """登记 V → 首次碰 V → 以 result 结束；返回登记回执。"""
    got = led.register_v(market, start, end, tid)
    assert got["registered"] and got["verdict"] == PROTOCOL_OK, got["reasons"]
    assert led.touch_v(tid)["verdict"] == PROTOCOL_OK
    led.finish_v(tid, result)
    return got


def test_v_can_be_reused_and_counts_tasks():
    """B16（v007.1）：一段 V 可以给多个任务用——用过的块照登记，记这段 V 被第几个任务用；只加注，不判不可认证。"""
    led = BlockLedger()
    first = _run_v(led, "t1")
    assert len(first["blocks"]) == 12 and first["blocks"][0] == "2023-01"
    assert (first["n"], first["note"], first["reused_from"]) == (1, "", [])          # 第一个任务：不加注
    assert set(first["prior_uses"].values()) == {0}
    again = led.register_v(MKT, "2023-06-15", "2024-03-31", "t2")                    # 只重叠半年
    assert again["registered"] and again["verdict"] == PROTOCOL_OK and again["reasons"] == []
    assert again["n"] == 2 and again["reused_from"] == ["t1"]
    assert again["note"] == REUSE_NOTE.format(n=2) == "这段 V 给过 2 个任务：闸门结论更只能当参考"
    assert again["prior_uses"]["2023-06"] == 1 and again["prior_uses"]["2024-03"] == 0
    assert "照登记" in again["warnings"][0] and "不判不可认证" in again["warnings"][0]
    chk = led.touch_v("t2")
    assert chk["verdict"] == PROTOCOL_OK and (chk["n"], chk["note"]) == (2, again["note"])
    third = _run_v(led, "t3")                                   # 2023-06…12 给过 t1、t2 两个 → 按块取最大，第 3 个
    assert third["n"] == 3 and third["reused_from"] == ["t1", "t2"] and third["prior_uses"]["2023-01"] == 1
    st = led.status(MKT, V0, V1)
    assert st["state"] == USED and st["used_by"] == ["t1", "t2", "t3"] and st["n_uses"] == 3
    assert st["block_uses"]["2023-01"] == 2 and st["block_uses"]["2023-12"] == 3
    b = led.block_status(MKT, "2023-07")
    assert (b["state"], b["used_by"], b["n_uses"]) == (USED, ["t1", "t2", "t3"], 3)
    assert led.check_v("t1")["n"] == 1                                               # 后来的任务不改先来者的名次
    us = led.register_v("US", V0, V1, "t4")                                          # 别的市场不相干
    assert us["registered"] and us["n"] == 1
    assert led.block_status(MKT, "2024-04")["state"] == UNUSED
    assert led.blocks("2023-06-15", "2023-08-02") == ["2023-06", "2023-07", "2023-08"]   # 盖住半个块也算
    with pytest.raises(ValueError, match="block_freq"):
        BlockLedger(block_freq="不是周期")


def test_running_task_counts_too_and_task_id_is_unique():
    led = BlockLedger()
    led.register_v(MKT, V0, V1, "a")                                    # 登记了、还没结束
    b = led.register_v(MKT, V0, V1, "b")
    assert b["registered"] and b["n"] == 2                              # 还在跑的任务也算一个
    with pytest.raises(ValueError, match="登记过"):
        led.register_v(MKT, "2020-01-01", "2020-06-30", "a")             # 任务 ID 不能重用


def test_abort_before_touch_leaves_the_count_after_touch_stays():
    led = BlockLedger()
    led.register_v(MKT, V0, V1, "t1")
    out = led.abort_v("t1")                                             # 碰 V 前中止 → 退出计数，块退回未用
    assert out["refunded"] and out["state"] == UNUSED and led.status(MKT, V0, V1)["state"] == UNUSED
    chk = led.check_v("t1")                                             # 退回了，再拿它跑闸门不算数
    assert chk["verdict"] == UNCERTIFIABLE and chk["reasons"][0].startswith("登记被改") and chk["n"] is None
    assert led.register_v(MKT, V0, V1, "t2")["n"] == 1                  # 退出计数的不算一个
    led.touch_v("t2")
    out = led.abort_v("t2")                                             # 碰 V 后中止 → 照计、保持已用
    assert not out["refunded"] and out["state"] == USED and led.status(MKT, V0, V1)["state"] == USED
    assert led.register_v(MKT, V0, V1, "t3")["n"] == 2


def test_earlier_abort_before_touch_lowers_later_n():
    """排在前面的任务碰 V 前中止（没看过 V）→ 它退出计数，后来者的 n 跟着减（以账本重放为准）。"""
    led = BlockLedger()
    led.register_v(MKT, V0, V1, "a")
    assert led.register_v(MKT, V0, V1, "b")["n"] == 2
    led.abort_v("a")
    chk = led.check_v("b")
    assert chk["verdict"] == PROTOCOL_OK and (chk["n"], chk["note"], chk["reused_from"]) == (1, "", [])
    assert led.block_status(MKT, "2023-05")["used_by"] == ["b"]


def test_finish_needs_touch_and_three_valued_result():
    led = BlockLedger()
    led.register_v(MKT, V0, V1, "t1")
    with pytest.raises(ValueError, match="abort_v"):
        led.finish_v("t1", NOT_PASSED)
    led.touch_v("t1")
    with pytest.raises(ValueError, match="只能是"):
        led.finish_v("t1", 1.23)                                        # V 段数值进不了账本
    done = led.finish_v("t1", PASSED)
    assert done["state"] == USED and (done["n"], done["note"]) == (1, "")
    with pytest.raises(ValueError, match="已经"):
        led.abort_v("t1")
    assert led.check_v("t1")["verdict"] == PROTOCOL_OK
    missing = led.check_v("没有这个任务")                                  # V 没在账本登记 → 归「登记被改」
    assert missing["verdict"] == UNCERTIFIABLE and missing["reasons"][0].startswith("登记被改") and missing["n"] is None


def test_ledger_file_is_append_only_shared_and_replayed(tmp_path):
    """B9 / B16：写进文件、只追加、跨任务共用、每行带任务 ID；重开同一个文件照文件重放。"""
    path = tmp_path / "blocks.jsonl"
    a = BlockLedger(path)
    _run_v(a, "t1")
    a.start_exam(MKT, T0, T1, demand_id="d1", task_id="t1", cand_id="c1")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["event"] for x in lines] == ["登记V", "碰V", "结束V", "开考"]
    assert all(json.loads(x)["task_id"] == "t1" for x in lines)
    b = BlockLedger(path)                                               # 另一个任务（另一个进程）打开同一本账
    assert b.status(MKT, V0, V1)["state"] == USED and b.status(MKT, T0, T1)["state"] == EXAMINED
    assert b.register_v(MKT, V0, V1, "t2")["n"] == 2                     # 用过的 V 照登记，记第 2 个任务
    assert b.register_v(MKT, "2022-01-01", "2022-12-31", "t3")["n"] == 1
    assert a.status(MKT, V0, V1)["used_by"] == ["t1", "t2"]             # 每次查账都重读文件
    assert a.check_v("t2")["n"] == 2
    assert path.read_text(encoding="utf-8").splitlines()[:4] == lines         # 旧行不动，只往后加


def test_race_for_same_v_counts_in_file_order(tmp_path):
    """两个进程同时登记同一段 V（查账时都还没看到对方）：两个都照登记，按文件里的先后排第几个（B16，v007.1）。"""
    path = tmp_path / "blocks.jsonl"
    a = BlockLedger(path)
    assert a.register_v(MKT, V0, V1, "a")["n"] == 1
    jsonl_append(path, {"event": "登记V", "task_id": "b", "market": MKT, "start": V0, "end": V1,
                        "blocks": a.blocks(V0, V1)})                    # b 抢跑：没看到 a 就写了自己的登记行
    first = a.check_v("a")
    assert first["verdict"] == PROTOCOL_OK and first["n"] == 1
    late = BlockLedger(path).check_v("b")
    assert late["verdict"] == PROTOCOL_OK and late["n"] == 2 and late["note"] == REUSE_NOTE.format(n=2)


def test_truncated_last_line_is_skipped_and_next_line_starts_fresh(tmp_path):
    """断电时写了一半的残行：读的时候跳过；下一行另起一行，不和残行粘在一起。"""
    path = tmp_path / "blocks.jsonl"
    BlockLedger(path).register_v(MKT, V0, V1, "t1")
    with open(path, "ab") as f:
        f.write('{"event": "碰V", "task_id": "t1"'.encode("utf-8")[:12])    # 断在一个汉字中间
    led = BlockLedger(path)
    assert led.status(MKT, V0, V1)["state"] == USED
    led.touch_v("t1")
    assert [r["event"] for r in jsonl_rows(path)] == ["登记V", "碰V"]
    assert led.abort_v("t1")["state"] == USED                           # 碰过 V：中止也不退回


def test_rerun_after_crash_resumes_instead_of_counting_twice(tmp_path):
    """断了重跑：同一任务、同一窗口再登记，照原登记接着用；同一场考试再报到不算再考；都不另写行。"""
    path = tmp_path / "blocks.jsonl"
    BlockLedger(path).register_v(MKT, V0, V1, "t1")
    led = BlockLedger(path)                                             # 机器重启后
    again = led.register_v(MKT, V0, V1, "t1")
    assert again["registered"] and again["resumed"] and again["verdict"] == PROTOCOL_OK and again["n"] == 1
    with pytest.raises(ValueError, match="登记过"):
        led.register_v(MKT, "2022-01-01", "2022-12-31", "t1")           # 同一任务换窗口不行
    assert led.touch_v("t1")["verdict"] == PROTOCOL_OK
    first = led.start_exam(MKT, T0, T1, demand_id="d1", task_id="t1", cand_id="c1")
    rerun = BlockLedger(path).start_exam(MKT, T0, T1, demand_id="d1", task_id="t1", cand_id="c1")
    assert rerun["allowed"] and rerun["resumed"] and (rerun["k"], rerun["note"]) == (first["k"], first["note"])
    other = led.start_exam(MKT, T0, T1, demand_id="d1", task_id="t1", cand_id="c2")    # 换了候选就是再考
    assert not other["allowed"] and other["verdict"] == UNCERTIFIABLE
    assert [r["event"] for r in jsonl_rows(path)] == ["登记V", "碰V", "开考"]


# ---------------- 测试段（A2）----------------

def test_exam_once_per_demand_and_retest_note():
    """A2：每条需求考一次，同一需求再考 = 不可认证；同一段可以被别的需求再考，从第 2 次起加注，不设闸。"""
    led = BlockLedger()
    first = led.start_exam(MKT, T0, T1, demand_id="d1", task_id="t1", cand_id="c1")
    assert first["allowed"] and (first["n"], first["k"], first["note"], first["confirmed_by"]) == (0, 1, "", BY_HUMAN)
    assert led.status(MKT, T0, T1)["state"] == EXAMINED
    s2, e2 = "2024-06-01", "2025-03-31"                                 # 与第一场相交 = 同一段
    again = led.start_exam(MKT, s2, e2, demand_id="d1", task_id="t9", cand_id="c9")
    assert not again["allowed"] and again["verdict"] == UNCERTIFIABLE
    second = led.start_exam(MKT, s2, e2, demand_id="d2", task_id="t2", cand_id="c2", confirmed_by=BY_AI)
    assert second["allowed"] and second["verdict"] == PROTOCOL_OK
    assert (second["n"], second["k"], second["note"]) == (1, 2, RETEST_NOTE)
    assert led.block_status(MKT, "2024-07")["demands"] == ["d1", "d2"] and led.status(MKT, T0, T1)["n_exams"] == 2
    assert led.block_status(MKT, "2025-02")["n_exams"] == 1
    later = led.start_exam(MKT, "2026-01-01", "2026-03-31", demand_id="d1", task_id="t3", cand_id="c3")
    assert later["allowed"]                                              # 不相交的新一段，不算再考
    assert [r["confirmed_by"] for r in led.rows() if r["event"] == "开考"] == [BY_HUMAN, BY_AI, BY_HUMAN]


def test_exam_takes_one_confirmed_candidate():
    led = BlockLedger()
    with pytest.raises(ValueError, match="一个候选"):
        led.start_exam(MKT, T0, T1, demand_id="d1", task_id="t1", cand_id=["c1", "c2"])
    with pytest.raises(ValueError, match="确认人"):
        led.start_exam(MKT, T0, T1, demand_id="d1", task_id="t1", cand_id="c1", confirmed_by="没人")
    assert led.status(MKT, T0, T1)["state"] == UNUSED


def test_examined_blocks_can_be_used_as_v_with_a_warning():
    """B19：被考过的块照用、不判不可认证（回执里列出来）；当过 V 之后状态记「已用」。"""
    led = BlockLedger()
    led.start_exam(MKT, "2023-03-01", "2023-04-30", demand_id="d0", task_id="x0", cand_id="c0")
    got = led.register_v(MKT, V0, V1, "t1")
    assert got["registered"] and got["verdict"] == PROTOCOL_OK and got["examined_blocks"] == ["2023-03", "2023-04"]
    assert "不判不可认证" in got["warnings"][0] and got["n"] == 1     # 考过不算「给过任务当 V」
    assert led.check_v("t1")["verdict"] == PROTOCOL_OK
    st = led.block_status(MKT, "2023-03")
    assert (st["state"], st["used_by"], st["n_uses"], st["n_exams"]) == (USED, ["t1"], 1, 1)


# ---------------- 旧 B19 的 V 段定位（v007.1 起默认流程不调用；留给按旧规矩挪过 V 的登记复核）----------------

LOC_DATES = pd.bdate_range("2018-01-01", "2024-12-31")
LOC_SPLIT = split_segments(LOC_DATES)


def _locate(led, **kw):
    return led.locate_v(MKT, LOC_DATES, t_val=252, test_start=LOC_SPLIT.t_1, d_pur=LOC_SPLIT.d_pur, **kw)


def test_locator_fresh_ledger_finds_natural_window():
    got = _locate(BlockLedger())
    assert got["found"] and (got["start"], got["end"]) == (LOC_SPLIT.v.start, LOC_SPLIT.v.end)


def test_locator_skips_used_and_examined_blocks():
    led = BlockLedger()
    sp = LOC_SPLIT
    _run_v(led, "t1", sp.v.start, sp.v.end, result=PASSED)              # 默认 V 窗已用
    first = _locate(led)
    assert first["found"] and first["end"] == pd.Timestamp("2022-12-30")    # 往过去挪到上一个干净块
    led.start_exam(MKT, "2022-06-01", "2022-08-31", demand_id="d0", task_id="x0", cand_id="c0")
    got = _locate(led)
    assert got["end"] == pd.Timestamp("2022-05-31")                       # 被考过的块不算干净
    assert not set(got["blocks"]) & {"2022-06", "2022-07", "2022-08"}


def test_locator_reports_no_clean_window_with_three_options():
    led = BlockLedger()
    led.start_exam(MKT, "2018-01-01", "2022-06-30", demand_id="d0", task_id="x0", cand_id="c0")
    _run_v(led, "t1", LOC_SPLIT.v.start, LOC_SPLIT.v.end, result=PASSED)
    got = _locate(led, n_tests=5)
    assert not got["found"] and got["message"] == NO_CLEAN_V
    longest = len(pd.bdate_range("2022-07-01", "2022-12-30"))
    assert got["longest_clean"] == longest
    assert got["options"][0].startswith("缩短 V") and f"SR_mde = {G.sr_mde(5, longest):.2f}" in got["options"][0]
    assert got["options"][1].startswith("照用被考过的块（不判不可认证）：") and "2022-12-30" in got["options"][1]
    assert got["options"][2] == "换市场"                                    # 例程不替人选
