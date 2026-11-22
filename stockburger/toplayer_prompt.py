# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""顶层提示：禁令句默认（C13）与上下文白名单（A4）。

一句话：顶层大模型的训练语料可能盖住测试期（它「记得」那段行情），所以给它的提示默认加一句「别用测试期行情」的
禁令句（C13，默认开、可关）；提示里放什么由脚本按白名单拼（A4，可选）——需求原文、字段说明（名称、定义式、频率、
覆盖期、缺失率、单位）、汇总表、新颖性标记，不放逐期数值。

规格：验收标准 v007 第七节 C13、第五节 A4。
- 禁令句按提示里有没有汇总表选：不带表（R0）用 W2，带表（R1+）用 W1；位置在任务句之后、作答格式之前。英文原句逐字照
  作者早先两轮探针实际发出的句子（W1、W2 各一句；探针 = 拿同一份问卷问顶层大模型、看它会不会用上测试期记忆的小实验），
  改措辞或位置算没测过的写法。效果没在真实环境验证过，不当防线；开着也照记审计账（records.py，C1）。
- 按提示类型选禁令句是写死的配置，不读任何读数，不算 C2 禁止的分支。
- 本模块只有 R0 提示（只见字段说明、不带表），它的签名就是白名单：没有放数据的位置。测试窗日期照印，因为两句禁令句
  都拿「a ticket's test window」说事。R1+ 提示接顶层前写定，不沿用 R0 的角色句「you never see any data values」
  （和汇总表矛盾）。
提示用英文，与探针同语种。
"""
from __future__ import annotations

import hashlib
import numbers
import re
from dataclasses import dataclass

import pandas as pd

# R0 提示的固定文字，逐字照探针问卷（改一个字就算没测过的写法）
ROLE_LINE = ('You are the planning model ("top layer") of an automated quantitative research system. '
             'Scripts, not you, fetch data, compute features, train and backtest. '
             'You see only the feature schema below; you never see any data values.')
SCHEMA_HEADER = "FEATURE-FAMILY SCHEMA (names, definitions, frequency only):"
# 字段说明里填了覆盖期、缺失率或单位时换这一行，免得表头说「只有频率」
SCHEMA_HEADER_FULL = ("FEATURE-FAMILY SCHEMA (names, definitions, frequency, coverage, missing rate, unit; "
                      "no per-period values):")
TASK_SENTENCE = ("TASK. There is one ticket: the client request above, with the test window listed below. "
                 "Rank the {n} families from highest priority (most likely to serve the client request over that "
                 "ticket's test window) to lowest.")

# ---------------------------------------------------------------------------
# 禁令句（C13）：不加 / W1 / W2，按提示类型选默认
# ---------------------------------------------------------------------------
# W1：前一轮探针实际发出的原句；带表提示（R1+）的默认
W1_SENTENCE = ("Rule: do not use any knowledge you may have of how markets, sectors or these factors actually "
               "behaved during or after a ticket's test window. Base each ranking only on what could have been "
               "known before the window starts.")
# W2：后一轮探针实际发出的原句（先划出能用的，再禁测试窗内及之后）；不带表提示（R0）的默认
W2_SENTENCE = ("Rule: you may use the pre-window summary table (if one is given) and anything that was publicly known "
               "before a ticket's test window starts, including how markets and these factors had behaved up to that "
               "date. Do not use any knowledge you may have of how markets, sectors or these factors actually behaved "
               "during or after the ticket's test window.")
BAN_VARIANTS = ("off", "W1", "W2")
BAN_SENTENCES = {"off": None, "W1": W1_SENTENCE, "W2": W2_SENTENCE}
PROMPT_TYPES = ("R0", "R1+")                   # 不带汇总表 / 带汇总表（只按有没有表分）
DEFAULT_BAN = {"R0": "W2", "R1+": "W1"}

_NAME_OK = re.compile(r"[^\s|>]+")             # 族名进作答格式「A > B」，不能带空白、「|」「>」


def _check_choice(name: str, value, allowed) -> str:
    if value not in allowed:
        raise ValueError(f"{name} 只能是 {' / '.join(allowed)}，收到 {value!r}")
    return value


def _check_flag(value, name: str = "ban") -> bool:
    if value not in (True, False):
        raise ValueError(f"{name} 只能是 True / False，收到 {value!r}")
    return bool(value)


def sentence_hash(text: str | None) -> str | None:
    """句子哈希：UTF-8 编码后的 sha256 十六进制串（审计账记的就是它，C1）；没有句子（禁令关）返回 None。"""
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_type_of(has_table: bool) -> str:
    """提示类型：没有汇总表 = R0，有表 = R1+；不看第几轮。"""
    return "R1+" if _check_flag(has_table, "has_table") else "R0"


def default_ban_variant(prompt_type: str, ban: bool = True) -> str:
    """C13 的默认写法：禁令关 → off；开 → R0 用 W2、R1+ 用 W1（查表的静态配置，不算 C2 的分支）。"""
    _check_choice("prompt_type", prompt_type, PROMPT_TYPES)
    return DEFAULT_BAN[prompt_type] if _check_flag(ban) else "off"


@dataclass(frozen=True)
class PromptSpec:
    """一次顶层调用里要进审计账的提示配置（C1）：提示类型与禁令句写法；所用句子与句子哈希由写法查表得出。"""
    prompt_type: str      # R0 / R1+
    ban_variant: str      # off / W1 / W2

    def __post_init__(self):
        _check_choice("prompt_type", self.prompt_type, PROMPT_TYPES)
        _check_choice("ban_variant", self.ban_variant, BAN_VARIANTS)

    @property
    def ban(self) -> bool:
        return self.ban_variant != "off"

    @property
    def ban_sentence(self) -> str | None:
        return BAN_SENTENCES[self.ban_variant]

    @property
    def ban_hash(self) -> str | None:
        return sentence_hash(self.ban_sentence)

    def to_json(self) -> dict:
        return {"prompt_type": self.prompt_type, "ban_variant": self.ban_variant, "ban_hash": self.ban_hash}


# ---------------------------------------------------------------------------
# 上下文白名单（A4）：字段说明
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FamilySchema:
    """一族特征的字段说明（A4）：名称、定义式、频率、覆盖期、缺失率、单位。

    只收说明，不收逐期数值：文字项只收 str；缺失率收一个 0–1 之间的数或现成的文字；序列、数组一律不收。
    范围（如 US large and mid caps）写进定义式，与探针问卷的行同格式：「NAME: 定义; 范围; 频率.」。
    """
    name: str
    definition: str
    frequency: str = "daily"
    coverage: str = ""
    missing_rate: float | str | None = None
    unit: str = ""

    def __post_init__(self):
        for key in ("name", "definition", "frequency", "coverage", "unit"):
            value = getattr(self, key)
            if not isinstance(value, str):
                raise TypeError(f"字段说明的 {key} 只收文字，收到 {type(value).__name__}（不放逐期数值，A4）")
        mr = self.missing_rate
        if mr is not None and not isinstance(mr, str):
            if isinstance(mr, bool) or not isinstance(mr, numbers.Real):
                raise TypeError(f"缺失率只收一个数（0–1）或文字，收到 {type(mr).__name__}（不放逐期数值，A4）")
            if not 0.0 <= float(mr) <= 1.0:
                raise ValueError(f"缺失率要在 0–1 之间，收到 {mr!r}")
        if not _NAME_OK.fullmatch(self.name):
            raise ValueError(f"族名要非空，且不含空白、「|」「>」（作答格式要用），收到 {self.name!r}")
        if not self.definition.strip():
            raise ValueError(f"族 {self.name} 缺定义")

    @property
    def has_extras(self) -> bool:
        """填了覆盖期、缺失率或单位（表头要换成 SCHEMA_HEADER_FULL）。"""
        return bool(self.coverage.strip() or self.unit.strip() or self.missing_rate is not None)

    def line(self) -> str:
        """提示里的一行：「NAME: 定义; 频率[; coverage …][; missing rate …][; unit …].」，空着的项不印。"""
        mr = self.missing_rate
        rate = "" if mr is None else f"missing rate {mr if isinstance(mr, str) else format(float(mr), '.1%')}"
        extras = (f"coverage {self.coverage}" if self.coverage.strip() else "", rate,
                  f"unit {self.unit}" if self.unit.strip() else "")
        parts = [p.strip().rstrip(".;").strip() for p in (self.definition, self.frequency, *extras)]
        return f"{self.name}: {'; '.join(p for p in parts if p)}."


def _iso(x) -> str:
    return pd.Timestamp(x).date().isoformat()


def build_schema_only_prompt(demand_text: str, families, test_window, *, ban: bool = True) -> tuple[str, PromptSpec]:
    """生成 R0 提示（只见字段说明、不带表），连同要进审计账的提示配置一起返回。

    demand_text：需求原文，照抄进 CLIENT REQUEST（人的原话在白名单里，里面的数字照留）；
    families：FamilySchema（或同字段的 dict）序列，至少一个、族名不重复；
    test_window：(t_1, t_2) 测试窗首末日，只印日期。
    ban=True（默认）在任务句之后、作答格式之前附 W2 原句；ban=False 只是少这一行，别的一字不变。
    """
    if not isinstance(demand_text, str) or not demand_text.strip():
        raise ValueError("需求原文不能为空")
    fams = [f if isinstance(f, FamilySchema) else FamilySchema(**f) for f in families]
    if not fams:
        raise ValueError("至少要一个特征族")
    names = [f.name for f in fams]
    if len(set(names)) != len(names):
        raise ValueError(f"族名重复：{names}")
    t_1, t_2 = (_iso(x) for x in test_window)
    if t_1 > t_2:
        raise ValueError(f"测试窗首日 {t_1} 晚于末日 {t_2}")
    spec = PromptSpec(prompt_type_of(False), default_ban_variant("R0", ban))
    header = SCHEMA_HEADER_FULL if any(f.has_extras for f in fams) else SCHEMA_HEADER
    lines = [ROLE_LINE, "", f'CLIENT REQUEST: "{demand_text.strip()}"', "", header]
    lines += [f.line() for f in fams]
    lines += ["", TASK_SENTENCE.format(n=len(fams))]
    if spec.ban_sentence is not None:                 # 位置同探针：任务句之后、作答格式之前
        lines.append(spec.ban_sentence)
    lines += ["Do not use any tools. Answer with one line per ticket and nothing else, exactly in this format:",
              "T## | " + " > ".join(["FAMILY"] * len(fams)) + " | reason in at most 12 words",
              "Tickets:",
              f"T01 | test window {t_1} to {t_2}"]
    return "\n".join(lines) + "\n", spec


def schema_only_prompt(demand_text: str, families, test_window, *, ban: bool = True) -> str:
    """只要提示文字时用；要记审计账就用 build_schema_only_prompt，它多返回一份 PromptSpec。"""
    return build_schema_only_prompt(demand_text, families, test_window, ban=ban)[0]
