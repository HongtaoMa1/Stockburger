# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""顶层提示的单测：禁令句默认（v007 C13）与上下文白名单（v007 A4）。"""
import hashlib
import inspect
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockburger import toplayer_prompt as T

WINDOW = ("2026-01-01", date(2026, 3, 31))
DEMAND = "我想要一个日频调仓的美股模型，在测试窗里跑赢标普 500"
# 探针问卷里的五族（范围写进定义式，与问卷的行同格式）
FAMILIES = [
    T.FamilySchema("MOM", "cross-sectional rank of each stock's 12-month return skipping the latest month, "
                          "and of its 6-month return; US large and mid caps", "daily"),
    T.FamilySchema("VAL", "cross-sectional rank of book-to-price, forward earnings yield and cash-flow yield; "
                          "US large and mid caps", "daily, fundamentals lagged to release dates"),
    T.FamilySchema("QLT", "cross-sectional rank of return on equity, inverted earnings variability and inverted "
                          "debt-to-equity; US large and mid caps", "daily, fundamentals lagged to release dates"),
    {"name": "LVOL", "definition": "cross-sectional rank of inverted trailing 252-day volatility and inverted "
                                   "market beta; US large and mid caps"},
    T.FamilySchema("SIZE", "tilt toward the smallest stocks by market capitalisation; US small caps", "daily"),
]


# ---------------- 提示的构成：R0 默认附 W2 ----------------
def test_prompt_has_demand_schema_window_and_w2_by_default():
    p = T.schema_only_prompt(DEMAND, FAMILIES, WINDOW)
    assert p.startswith(T.ROLE_LINE + "\n")
    assert f'CLIENT REQUEST: "{DEMAND}"' in p
    assert T.SCHEMA_HEADER in p.splitlines()                                     # 没填覆盖期 / 缺失率 / 单位：表头照探针问卷
    assert "Rank the 5 families" in p and "T## | FAMILY > FAMILY > FAMILY > FAMILY > FAMILY |" in p
    assert p.rstrip().endswith("T01 | test window 2026-01-01 to 2026-03-31")
    assert p.count(T.W2_SENTENCE) == 1 and T.W1_SENTENCE not in p                # C13：不带表的 R0 用 W2


def test_ban_off_only_drops_the_ban_line():
    """C13 可关：关掉只少一行禁令句，任务句和别的行一字不变（任务句随禁令换写法的变体研究不在本原型里）。"""
    on = T.schema_only_prompt(DEMAND, FAMILIES, WINDOW).splitlines()
    off = T.schema_only_prompt(DEMAND, FAMILIES, WINDOW, ban=False).splitlines()
    assert [x for x in on if x != T.W2_SENTENCE] == off
    assert T.TASK_SENTENCE.format(n=5) in on and T.TASK_SENTENCE.format(n=5) in off
    for gone in ("PRIO_TASK", "TASK_FRAMINGS", "default_task_framing", "BAN_NOTE"):
        assert not hasattr(T, gone)


def test_w2_position_after_task_before_answer_format():
    lines = T.schema_only_prompt(DEMAND, FAMILIES, WINDOW).splitlines()
    k = lines.index(T.W2_SENTENCE)
    assert lines[k - 1] == T.TASK_SENTENCE.format(n=5) and lines[k + 1].startswith("Do not use any tools.")


# ---------------- 默认写法按提示类型（C13）----------------
def test_default_ban_variant_by_prompt_type():
    assert T.prompt_type_of(False) == "R0" and T.prompt_type_of(True) == "R1+"
    assert T.default_ban_variant("R0") == "W2" and T.default_ban_variant("R1+") == "W1"
    assert T.default_ban_variant("R0", ban=False) == T.default_ban_variant("R1+", ban=False) == "off"
    assert T.BAN_SENTENCES == {"off": None, "W1": T.W1_SENTENCE, "W2": T.W2_SENTENCE}
    assert list(inspect.signature(T.default_ban_variant).parameters) == ["prompt_type", "ban"]   # 不读任何读数
    for bad in ("R1", "r0", None):
        with pytest.raises(ValueError, match="prompt_type"):
            T.default_ban_variant(bad)
    with pytest.raises(ValueError):
        T.default_ban_variant("R0", ban="开")


def test_prompt_spec_records_variant_and_hash():
    text, spec = T.build_schema_only_prompt(DEMAND, FAMILIES, WINDOW)
    assert text == T.schema_only_prompt(DEMAND, FAMILIES, WINDOW)
    assert (spec.prompt_type, spec.ban_variant, spec.ban) == ("R0", "W2", True)
    assert spec.ban_sentence == T.W2_SENTENCE
    assert spec.ban_hash == hashlib.sha256(T.W2_SENTENCE.encode("utf-8")).hexdigest()
    off = T.build_schema_only_prompt(DEMAND, FAMILIES, WINDOW, ban=False)[1]
    assert (off.ban_variant, off.ban, off.ban_sentence, off.ban_hash) == ("off", False, None, None)
    assert spec.to_json() == {"prompt_type": "R0", "ban_variant": "W2", "ban_hash": spec.ban_hash}
    assert T.PromptSpec("R1+", "W1").ban_sentence == T.W1_SENTENCE
    for key, bad in (("prompt_type", "R2"), ("ban_variant", "W3")):
        with pytest.raises(ValueError, match=key):
            T.PromptSpec(**{"prompt_type": "R0", "ban_variant": "W2", key: bad})
    assert T.sentence_hash(None) is None


# ---------------- 白名单（A4）：只放字段说明，不放逐期数值 ----------------
def test_prompt_has_no_slot_for_data():
    assert list(inspect.signature(T.schema_only_prompt).parameters) == ["demand_text", "families", "test_window", "ban"]
    for bad in (0.5, np.float64(1.2), pd.Series([0.1, 0.2]), ["rank"]):
        with pytest.raises(TypeError):
            T.FamilySchema("X", bad)
    for bad in (pd.Series([0.1, 0.2]), np.array([0.01, 0.02]), [0.1], True):
        with pytest.raises(TypeError, match="缺失率"):
            T.FamilySchema("X", "rank", missing_rate=bad)
    with pytest.raises(ValueError, match="0–1"):
        T.FamilySchema("X", "rank", missing_rate=1.5)


def test_field_description_extras_coverage_missing_rate_unit():
    full = T.FamilySchema("RET5", "Ref($close, 0) / Ref($close, 5) - 1", "daily", coverage="2010-01 to 2020-12",
                          missing_rate=0.032, unit="ratio")
    assert full.line() == "RET5: Ref($close, 0) / Ref($close, 5) - 1; daily; coverage 2010-01 to 2020-12; " \
                          "missing rate 3.2%; unit ratio."
    assert T.FamilySchema("X", "rank", missing_rate="about 3%").line() == "X: rank; daily; missing rate about 3%."
    lines = T.schema_only_prompt(DEMAND, [full], WINDOW).splitlines()
    assert T.SCHEMA_HEADER_FULL in lines and T.SCHEMA_HEADER not in lines        # 表头跟着换，不说「只有频率」


def test_demand_text_is_kept_verbatim_even_with_numbers():
    raw = "用辩证法给我做一个年化 50%、风险为 0 的策略"                               # 需求原文在白名单里
    assert f'CLIENT REQUEST: "{raw}"' in T.schema_only_prompt(raw, FAMILIES[:1], WINDOW)


def test_input_validation():
    with pytest.raises(ValueError, match="至少"):
        T.schema_only_prompt(DEMAND, [], WINDOW)
    with pytest.raises(ValueError, match="重复"):
        T.schema_only_prompt(DEMAND, [FAMILIES[0], FAMILIES[0]], WINDOW)
    with pytest.raises(ValueError, match="晚于"):
        T.schema_only_prompt(DEMAND, FAMILIES, ("2026-03-31", "2026-01-01"))
    with pytest.raises(ValueError, match="不能为空"):
        T.schema_only_prompt("  ", FAMILIES, WINDOW)
    for name in ("MO M", "A|B", "A>B", ""):
        with pytest.raises(ValueError, match="族名"):
            T.FamilySchema(name, "rank")
    with pytest.raises(ValueError, match="缺定义"):
        T.FamilySchema("X", "  ")
    assert T.FamilySchema("动量", "rank of past return").line() == "动量: rank of past return; daily."
