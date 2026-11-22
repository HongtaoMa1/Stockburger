# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""成品模块的单测：冻结（v007 C3）、冻结即可用（B25）、影子（C4）与日报（C5），记分口径照 B7。"""
import inspect
import math

import numpy as np
import pandas as pd
import pytest

from stockburger import product as P
from stockburger.records import STRESS_TAG

TAU = pd.Timestamp("2026-01-08 22:00")
STOCKS = [f"SH60000{k}" for k in range(8)]


def _product(pid="p1", w=(1.0, -0.5), trained_on="S"):
    return P.freeze(pid, TAU, weights=list(w), features=["RET5", "VOL20"], proposals=["量价因子做截面多空"],
                    trained_on=trained_on)


def _panel(dates, mu=0.002, seed=0, missing=0.0):
    """合成长表：预测 = 信号 + 噪声，标签 = mu·信号 + 噪声；missing 比例的格子留空（模拟成分股进出）。"""
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([pd.DatetimeIndex(dates), STOCKS], names=["datetime", "instrument"])
    sig = rng.normal(size=len(idx))
    pred = pd.Series(sig + rng.normal(scale=0.5, size=len(idx)), index=idx)
    label = pd.Series(mu * sig + rng.normal(scale=0.01, size=len(idx)), index=idx)
    if missing:
        label[rng.random(len(idx)) < missing] = np.nan
    return pred, label


# ---------------- 冻结（C3）----------------
def test_freeze_hashes_and_retrain_is_new_product():
    a, b = _product(), _product()
    assert a.fingerprint == b.fingerprint and len(a.weights_hash) == 64
    re = _product(w=(0.9, -0.5))                                           # 重训 = 新产物：哈希就变
    assert re.weights_hash != a.weights_hash and re.fingerprint != a.fingerprint
    swapped = P.freeze("p1", TAU, weights=[1.0, -0.5], features=["VOL20", "RET5"], proposals=["量价因子做截面多空"])
    assert swapped.features_hash != a.features_hash                        # 特征清单按顺序算
    assert _product(trained_on="S+V").fingerprint != a.fingerprint
    with pytest.raises(ValueError, match="trained_on"):
        _product(trained_on="测试")


def test_retrain_on_test_segment_is_a_usable_new_product():
    """C3：考完拿测试段数据重训 = 新产物——照样冻结、照样可用，日报不挂原来那次考试成绩。"""
    prod = _product()
    exam = P.exam_record(prod, *_panel(pd.bdate_range("2025-01-02", "2025-12-31")), segment=("2025-01-02", "2025-12-31"))
    new = _product(w=(0.8, -0.4), trained_on="S+V+测试")
    assert new.trained_on == "S+V+测试" and new.availability == P.AVAILABILITY
    assert P.FrozenProduct.from_json(new.to_json()) == new
    shadow = P.score_period(*_panel(pd.bdate_range("2026-01-09", periods=5)))
    rep = P.daily_report(new, shadow, as_of="2026-01-16", exam=exam)
    assert rep["lines"][1] == f"测试段成绩：{P.NOT_ATTACHED}" and rep["exam"] is None and rep["availability"] == "可用"


def test_fingerprint_rule_in_docstring_recomputes():
    """指纹算法写在 FrozenProduct.fingerprint 的文档串里，外人照着就能重算。"""
    import hashlib
    import json
    a = _product()
    blob = json.dumps([a.product_id, a.tau_freeze.isoformat(), a.weights_hash, a.features_hash, a.proposals_hash,
                       a.trained_on], sort_keys=True, ensure_ascii=False)
    assert hashlib.sha256(blob.encode("utf-8")).hexdigest() == a.fingerprint


def test_weights_hash_accepts_model_text_file_bytes_and_booster(tmp_path):
    class FakeBooster:                                                     # LightGBM Booster 的样子
        def model_to_string(self):
            return "tree\nversion=v4\n"
    path = tmp_path / "model.txt"
    path.write_text("tree\nversion=v4\n", encoding="utf-8", newline="\n")
    h = P.weights_hash("tree\nversion=v4\n")
    assert P.weights_hash(FakeBooster()) == P.weights_hash(path) == P.weights_hash(b"tree\nversion=v4\n") == h
    assert P.weights_hash(np.array([1.0, -0.5])) == P.weights_hash([1.0, -0.5]) != P.weights_hash([[1.0, -0.5]])
    prod = P.freeze("p1", TAU, weights=FakeBooster(), features=["RET5"], proposals=[])
    assert prod.same_weights(path) and not prod.same_weights("tree\nversion=v5\n")


def test_frozen_record_round_trips_and_detects_edits():
    a = _product()
    row = a.to_json()
    assert set(row) >= {"tau_freeze", "weights_hash", "features_hash", "proposals_hash", "fingerprint"}
    assert P.FrozenProduct.from_json(row) == a
    with pytest.raises(ValueError, match="记录被改过"):
        P.FrozenProduct.from_json({**row, "weights_hash": "0" * 64})


# ---------------- 记分口径（B7）----------------
def test_daily_scores_hand_computed():
    day = pd.Timestamp("2026-01-09")
    idx = pd.MultiIndex.from_product([[day], ["A", "B", "C", "D"]], names=["datetime", "instrument"])
    pred = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    label = pd.Series([0.01, -0.02, 0.03, 0.05], index=idx)
    out = P.daily_scores(pred, label)
    # 名次去均值 [-1.5, -0.5, 0.5, 1.5]，Σ|w| / 2 = 2 → 权重 [-0.75, -0.25, 0.25, 0.75]（多头 +1、空头 −1）
    assert out.loc[day, "ret"] == pytest.approx(-0.75 * 0.01 + 0.25 * 0.02 + 0.25 * 0.03 + 0.75 * 0.05)
    assert out.loc[day, "ic"] == pytest.approx(0.8)                       # 标签名次 [2, 1, 3, 4] 对预测名次 [1, 2, 3, 4]
    assert out.loc[day, "n"] == 4


def test_daily_scores_skips_missing_stocks_and_accepts_wide_and_swapped_index():
    dates = pd.bdate_range("2026-01-09", periods=30)
    pred, label = _panel(dates, missing=0.2)
    long = P.daily_scores(pred, label)
    assert (long["n"] < len(STOCKS)).any()                                # 缺的股票当天不进截面
    wide = P.daily_scores(pred.unstack(), label.unstack())
    pd.testing.assert_frame_equal(long, wide)
    swapped = P.daily_scores(pred.swaplevel().sort_index(), label.to_frame("LABEL0"))   # D.features 的 (instrument, datetime)
    pd.testing.assert_frame_equal(long, swapped)


def test_same_b7_numbers_as_screen_even_with_gaps():
    """B7 只有一个口径：长表摊成宽表后，逐日数与 screen.py 的同名函数逐格相同（缺值的格子也对得上）。"""
    from stockburger import screen
    dates = pd.bdate_range("2026-01-09", periods=20)
    pred, label = _panel(dates, seed=3, missing=0.3)
    f, y = pred.unstack().to_numpy(), label.unstack().to_numpy()
    out = P.daily_scores(pred.sample(frac=1.0, random_state=0), label)     # 行序打乱也一样
    np.testing.assert_array_equal(out["ret"].to_numpy(), screen.strategy_returns(f, y))
    np.testing.assert_array_equal(out["ic"].to_numpy(), screen.daily_rank_ic(f, y))
    assert (out["n"].to_numpy() == (np.isfinite(f) & np.isfinite(y)).sum(axis=1)).all()


def test_score_period_formulas():
    dates = pd.bdate_range("2026-01-09", periods=60)
    pred, label = _panel(dates, seed=1)
    s = P.score_period(pred, label)
    d = P.daily_scores(pred, label)
    assert s.n_days == 60 and (s.first, s.last) == ("2026-01-09", dates[-1].date().isoformat())
    assert s.ic_mean == pytest.approx(d["ic"].mean())
    assert s.ic_t == pytest.approx(d["ic"].mean() / (d["ic"].std(ddof=1) / math.sqrt(60)))
    assert s.ir == pytest.approx(d["ret"].mean() / d["ret"].std(ddof=1) * math.sqrt(252))
    empty = P.score_period(pred.iloc[:0], label.iloc[:0])
    assert (empty.n_days, empty.ic_mean, empty.ic_t, empty.ir, empty.first) == (0, None, None, None, None)
    one = P.score_period(pred.loc[[dates[0]]], label.loc[[dates[0]]])
    assert one.n_days == 1 and one.ic_t is None and one.ir is None and "—" in one.text()


# ---------------- 影子（C4）----------------
def test_shadow_only_counts_days_after_freeze_and_up_to_as_of():
    dates = pd.bdate_range("2025-12-01", "2026-04-30")
    pred, label = _panel(dates, seed=2)
    prod = _product()
    s = P.shadow_score(prod, pred, label, "2026-03-31")
    after = dates[(dates > TAU) & (dates <= pd.Timestamp("2026-03-31"))]
    assert s.n_days == len(after) and s.first == "2026-01-09" and s.last == "2026-03-31"   # 冻结当天的行情不进来
    junk = label.copy()
    junk[junk.index.get_level_values(0) <= TAU] = 9.9                      # 冻结前的数随便改，影子成绩不变
    assert P.shadow_score(prod, pred, junk, "2026-03-31") == s
    unrealized = label.copy()
    unrealized[unrealized.index.get_level_values(0) > pd.Timestamp("2026-03-27")] = np.nan   # 标签还没实现的日子留空
    assert P.shadow_score(prod, pred, unrealized, "2026-03-31").n_days == s.n_days - 2


# ---------------- 冻结即可用（B25）与日报（C5）----------------
def test_availability_never_changes_with_labels():
    prod = _product()
    dates = pd.bdate_range("2025-01-02", "2025-12-31")
    pred, label = _panel(dates)
    shadow = P.shadow_score(prod, *_panel(pd.bdate_range("2026-01-09", periods=10)), "2026-01-23")
    plain = P.exam_record(prod, pred, label, segment=("2025-01-02", "2025-12-31"))
    flagged = P.exam_record(prod, pred, label, segment=("2025-01-02", "2025-12-31"), ai_confirmed=True, nth_on_segment=2)
    retrained = _product(w=(0.3, 0.3))
    reports = [P.daily_report(prod, shadow, as_of="2026-01-23"),
               P.daily_report(prod, shadow, as_of="2026-01-23", exam=plain, L_label="clean"),
               P.daily_report(prod, shadow, as_of="2026-01-23", exam=flagged, L_label="contaminated"),
               P.daily_report(retrained, shadow, as_of="2026-01-23", exam=plain)]
    assert len({r["availability"].encode("utf-8") for r in reports}) == 1 and reports[0]["availability"] == "可用"
    assert prod.availability == retrained.availability == P.AVAILABILITY
    assert isinstance(inspect.getattr_static(P.FrozenProduct, "availability"), property)   # 没有收标签的口子


def test_daily_report_prints_exam_shadow_and_days_side_by_side():
    prod = _product()
    exam = P.exam_record(prod, *_panel(pd.bdate_range("2025-01-02", "2025-12-31"), seed=4),
                         segment=("2025-01-02", "2025-12-31"))
    shadow = P.shadow_score(prod, *_panel(pd.bdate_range("2026-01-09", periods=15), seed=5), "2026-03-31")
    rep = P.daily_report(prod, shadow, as_of="2026-03-31", exam=exam, L_label="partial",
                         extra_lines=["进考场前审查（D11）：前视特征 无；标签对齐 通过"])
    exam_line, shadow_line = rep["lines"][1], rep["lines"][2]
    assert exam_line.startswith("测试段成绩（2025-01-02 至 2025-12-31）：IC 均值 ") and f"［{STRESS_TAG}］" in exam_line
    assert shadow_line.startswith("影子成绩：已跑 15 个交易日；") and P.T_NOTE in shadow_line
    assert rep["lines"][3].startswith("进考场前审查") and rep["exam"]["fingerprint"] == prod.fingerprint
    assert rep["test_tag"] == STRESS_TAG and rep["shadow"]["n_days"] == 15
    clean = P.daily_report(prod, shadow, as_of="2026-03-31", exam=exam, L_label="clean")["lines"][1]
    assert "［" not in clean                                                # clean：不标
    flagged = P.exam_record(prod, *_panel(pd.bdate_range("2025-01-02", "2025-12-31"), seed=4),
                            segment=("2025-01-02", "2025-12-31"), ai_confirmed=True, nth_on_segment=2)
    line = P.daily_report(prod, shadow, as_of="2026-03-31", exam=flagged)["lines"][1]
    assert f"［{P.AI_CONFIRMED}］" in line and f"［{P.RESELECT_NOTE}］" in line
    assert P.daily_report(prod, shadow, as_of="2026-03-31")["lines"][1] == "测试段成绩：未考"


def test_retrained_product_does_not_carry_the_exam_score():
    prod = _product()
    exam = P.exam_record(prod, *_panel(pd.bdate_range("2025-01-02", "2025-12-31")), segment=("2025-01-02", "2025-12-31"))
    shadow = P.score_period(*_panel(pd.bdate_range("2026-01-09", periods=5)))
    rep = P.daily_report(_product(w=(0.2, 0.1)), shadow, as_of="2026-01-16", exam=exam)
    assert rep["lines"][1] == f"测试段成绩：{P.NOT_ATTACHED}" and rep["exam"] is None
    assert rep["availability"] == P.AVAILABILITY                            # 新产物照样可用


def test_exam_record_only_scores_inside_segment_and_round_trips():
    prod = _product()
    pred, label = _panel(pd.bdate_range("2024-10-01", "2026-01-30"), seed=6)
    exam = P.exam_record(prod, pred, label, segment=("2025-01-02", "2025-12-31"))
    inside = pd.bdate_range("2025-01-02", "2025-12-31")
    assert exam.score.n_days == len(inside) and (exam.score.first, exam.score.last) == ("2025-01-02", "2025-12-31")
    assert P.ExamRecord.from_json(exam.to_json()) == exam
    with pytest.raises(ValueError, match="晚于"):
        P.exam_record(prod, pred, label, segment=("2026-01-01", "2025-12-31"))
    with pytest.raises(ValueError, match="nth_on_segment"):
        P.exam_record(prod, pred, label, segment=("2025-01-02", "2025-12-31"), nth_on_segment=0)
