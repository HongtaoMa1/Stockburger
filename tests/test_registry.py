# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""试验登记与试验账本的单测（v007 B3 / B5 / B7；A2）。"""
import json

import numpy as np
import pandas as pd
import pytest

from stockburger.registry import ABANDONED, ABORTED, DONE, REJECTED, ROW_FIELDS, TrialRegistry
from stockburger.segments import PROTOCOL_OK, SEG_S, SEG_TEST, SEG_V, UNCERTIFIABLE


def _task_with_trials(path=None):
    reg = TrialRegistry("task-1", path)
    reg.record("c1", status=DONE, proposal_text="动量 20 日", scores={"S": 1.1})
    reg.record("c2", status=DONE, parent_id="c1")
    reg.record("c3", status=DONE)
    reg.record("c4", status=ABORTED)
    reg.record("c5", status=ABANDONED, parent_id="c3")
    reg.record("p1", status=REJECTED)
    reg.record("p2", status=REJECTED, proposal_text="Ref($close, -1)")
    return reg


def test_counts_n_t_n_prop_and_row_reconciliation():
    reg = _task_with_trials()
    assert reg.n_t == 5            # 含中止、被放弃
    assert reg.n_prop == 2         # 校验拒绝另计、不进 N_T
    assert len(reg.export_rows()) == reg.n_t + reg.n_prop
    s = reg.summary()
    assert s["verdict"] == PROTOCOL_OK and (s["N_T"], s["n_prop"], s["n_sent"]) == (5, 2, 0)
    assert s["n_fin"] is None and s["registration_hash"] is None


def test_abandoned_branch_all_counted():
    reg = TrialRegistry("t")
    reg.record("root", status=DONE)
    for k in range(4):   # 一整条分支中途放弃，训过的每个候选都计数
        reg.record(f"b{k}", status=ABANDONED, parent_id="root")
    assert reg.n_t == 5


def test_touching_v_before_registration_is_uncertifiable():
    reg = _task_with_trials()
    assert reg.enter_v("c1") == UNCERTIFIABLE
    reg.register(n_fin=5)           # 事后补登记也救不回来
    assert reg.verdict == UNCERTIFIABLE
    assert any("还没登记" in r for r in reg.problems())


def test_sending_more_than_n_fin_is_uncertifiable():
    reg = _task_with_trials()
    reg.register(n_fin=2)
    assert reg.enter_v("c1") == PROTOCOL_OK
    assert reg.enter_v("c2") == PROTOCOL_OK
    assert reg.enter_v("c2") == PROTOCOL_OK          # 重复报到不重复计数
    assert reg.n_sent == 2
    assert reg.enter_v("c3") == UNCERTIFIABLE
    assert any("超过登记的 n_fin" in r for r in reg.problems())
    assert reg.export_rows()[2]["touched"] == (SEG_S, SEG_V)


def test_record_with_v_touch_counts_as_entry():
    reg = TrialRegistry("t")
    reg.register(n_fin=1)
    reg.record("a", status=DONE, touched=(SEG_S, SEG_V))
    assert reg.verdict == PROTOCOL_OK and reg.n_sent == 1
    reg.record("b", status=DONE, touched=(SEG_S, SEG_V))
    assert reg.verdict == UNCERTIFIABLE


def test_registration_is_frozen():
    reg = TrialRegistry("t")
    h = reg.register(n_fin=5, r_max=1, alpha_gate=0.05, L=0)
    assert reg.register(n_fin=5, r_max=1, alpha_gate=0.05, L=0) == h    # 同样内容无事
    assert reg.verdict == PROTOCOL_OK
    assert reg.register(n_fin=50, r_max=1, alpha_gate=0.05, L=0) == h   # 改了：仍按原登记
    assert reg.verdict == UNCERTIFIABLE and any("登记被改" in r for r in reg.problems())
    assert reg.registration["n_fin"] == 5 and reg.summary()["n_fin"] == 5


def test_rounds_beyond_r_max_are_uncertifiable():
    reg = TrialRegistry("t")
    reg.register(n_fin=10, r_max=2)
    reg.record("x", status=DONE, round_no=2)
    assert reg.verdict == PROTOCOL_OK
    reg.record("y", status=DONE, round_no=3)           # 轮数超登记（S 段的轮次也算）
    assert reg.verdict == UNCERTIFIABLE and any("轮数超登记" in r for r in reg.problems())


def test_record_touching_test_segment_is_uncertifiable():
    reg = TrialRegistry("t")
    reg.record("a", status=DONE, touched=(SEG_S, SEG_TEST))
    assert reg.verdict == UNCERTIFIABLE and any("测试段" in r for r in reg.problems())


def test_rejected_proposals_never_touch_data():
    reg = TrialRegistry("t")
    with pytest.raises(ValueError, match="校验拒绝"):
        reg.record("p", status=REJECTED, touched=(SEG_S,))
    reg.record("p", status=REJECTED)
    with pytest.raises(ValueError):
        reg.enter_v("p")
    with pytest.raises(KeyError):
        reg.enter_v("nobody")
    with pytest.raises(ValueError):
        reg.record("q", status="成功")


def test_append_only():
    reg = _task_with_trials()
    with pytest.raises(ValueError, match="一个候选一行"):
        reg.record("c1", status=DONE)
    reg.append_info("c1", scores={"V": 0.4}, v_returns="paths/c1_v.parquet")
    with pytest.raises(ValueError, match="已写过"):
        reg.append_info("c1", scores={"S": 9.9})
    with pytest.raises(ValueError, match="已写过"):
        reg.append_info("c1", v_returns="other.parquet")
    row = reg.export_rows()[0]
    assert row["scores"] == {"S": 1.1, "V": 0.4} and row["v_returns"] == "paths/c1_v.parquet"


def test_export_rows_have_b7_fields_and_are_copies():
    reg = _task_with_trials()
    rows = reg.export_rows()
    assert set(rows[0]) == set(ROW_FIELDS)
    assert rows[0]["proposal_hash"] is not None and "动量" not in str(rows[0])   # 只存哈希不存原文
    assert rows[0]["task_id"] == "task-1" and rows[1]["parent_id"] == "c1" and rows[3]["status"] == ABORTED
    rows[0]["scores"]["S"] = -1.0
    rows[0]["status"] = ABANDONED
    fresh = reg.export_rows()[0]
    assert fresh["scores"]["S"] == 1.1 and fresh["status"] == DONE
    frame = reg.to_frame()
    assert isinstance(frame, pd.DataFrame) and list(frame.columns) == list(ROW_FIELDS) and len(frame) == 7


# ---------------- 落盘：只追加、跨任务共用、断了能接着记 ----------------

def test_ledger_file_replays_and_is_shared_across_tasks(tmp_path):
    path = tmp_path / "trials.jsonl"
    reg = _task_with_trials(path)
    reg.register(n_fin=2, r_max=1, alpha_gate=0.05)
    reg.enter_v("c1")
    other = TrialRegistry("task-2", path)                   # 另一个任务记在同一本账里
    other.record("c1", status=DONE)                         # 别的任务可以有同名候选
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 10 and all(x["task_id"] in ("task-1", "task-2") for x in lines)
    again = TrialRegistry("task-1", path)                   # 断了重开：照文件重放
    assert again.export_rows() == reg.export_rows()
    assert (again.n_t, again.n_prop, again.n_sent, again.verdict) == (5, 2, 1, PROTOCOL_OK)
    assert again.registration == reg.registration and again.registration_hash == reg.registration_hash
    assert "c1" in again and "c9" not in again               # 断了重跑时用来跳过已记的候选
    assert TrialRegistry("task-2", path).n_t == 1


def test_protocol_breaks_survive_reload(tmp_path):
    path = tmp_path / "trials.jsonl"
    reg = TrialRegistry("t", path)
    reg.record("a", status=DONE)
    reg.enter_v("a")                                         # 还没登记就送检
    reg.register(n_fin=5)
    reg.register(n_fin=50)                                   # 登记被改
    again = TrialRegistry("t", path)
    assert again.verdict == UNCERTIFIABLE and again.registration["n_fin"] == 5
    assert any("还没登记" in r for r in again.problems()) and any("登记被改" in r for r in again.problems())


def test_returns_round_trip_through_file(tmp_path):
    path = tmp_path / "trials.jsonl"
    s = pd.Series([0.01, np.nan, -0.02], index=pd.bdate_range("2020-01-01", periods=3))
    reg = TrialRegistry("t", path)
    reg.record("c1", status=DONE, s_returns=s)
    reg.append_info("c1", v_returns="cache/c1_v.parquet", scores={"V_闸门": "未过闸"})
    row = TrialRegistry("t", path).export_rows()[0]
    pd.testing.assert_series_equal(row["s_returns"], s.dropna(), check_freq=False, check_names=False)
    assert row["v_returns"] == "cache/c1_v.parquet" and row["scores"] == {"V_闸门": "未过闸"}
    arr = TrialRegistry("u").record("x", status=DONE, s_returns=np.array([0.1, np.nan]))["s_returns"]
    assert arr[0] == 0.1 and np.isnan(arr[1])               # 没有日期的数组照存，缺值还是 NaN


def test_truncated_last_line_is_skipped(tmp_path):
    path = tmp_path / "trials.jsonl"
    TrialRegistry("t", path).record("a", status=DONE)
    with open(path, "ab") as f:
        f.write('{"event": "试验", "task_id": "t", "cand_id": "b", "sta'.encode("utf-8"))   # 写到一半断电
    again = TrialRegistry("t", path)
    assert again.n_t == 1
    again.record("b", status=DONE)                           # 残行之后另起一行照常记
    assert TrialRegistry("t", path).n_t == 2
