# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""收单与拒单：需求原文 → 需求卡 → 处置「拒 / 问 / 转译」（v007 第八节）。

一句话：以拒结案的只有三种——要保证（F1）、明说非要亚秒级反应 / 托管机房 / 逐笔订单簿报撤单才能做（F2）、要做的市场
一个价格源都没有（F4）；衍生品只问不拒；其余要么问（发反问、等用户回话），要么转译（照做，但换成能做的写法并注明）。
全程按字段和规则确定地算，不调大模型。拿不准时取宽的读法：「高频」「逐笔」「延迟套利」单独出现只问不拒；卡上的市场
有一个接通就不拒；带否定的短语（「不要高频」「不做美股」「零风险是不可能的」）不进字段；用户不认字段读法就不拒。

两步：
1. extract_card(raw)：关键词替身——顶层接上前，按关键词把原文抽成需求卡，每个字段标「用户原文 / 顶层推断」和原文出处。
2. assess(card, boundary, waived, disputed)：规则脚本按卡算出全部命中与处置（同卡 + 同边界 ⇒ 同处置）。

规格：验收标准 v007 第八节 D1–D8 / D12、第五节 D9 / D10、第一节 A9、第九节「收单边界」。
命中表的 rule 列是条款号（D3 … D8、A9），code 列是原因码；拒单日志只追加。[GUESS] 数只能问、标、展示，不拿来拒单（A6）。

原因码速查（带 ′ ″ 的是同一条款下的变体；没有 F5，这个编号空着）：
- F1 拒：要零风险或保本，目标收益又高于 r_cert（D4）。F1′ 问：要保证，但目标不高于 r_cert、没写目标，或 r_cert 没登记。
- F2 拒：明说非要亚秒级反应 / 托管机房 / 逐笔订单簿报撤单才能做（D5）。F2′ 转译：机制换成不靠速度设施的慢版本。
  F2″ 问：「高频」「逐笔」「延迟套利」单独出现，问按日频做行不行。
- F3 问：用户点名衍生品（D6）。F3′ 转译：顶层推断的衍生品或价格敞口换成非衍生品候选。F3″ 问：点名的价格敞口只能用代理。
- F4 拒：要做的市场一个价格源都没有（D7）。F4′ 转译：点名特征拿不到，有代理就换、没有就删。F4″ 转译：卡上的市场有的
  接通、有的没接通，只做接通的，没接通的注明。
- F6 问：风险偏好和收益取向打架（D8）。F7 / F7′ 转译：给了目标收益、没给波动或回撤上限——有风险偏好或要保本就按它
  转译（F7），什么风险信息都没有就注明隐含的最低波动（F7′）。
- F8 问：点名的特征全都拿不到（D7）。
- F9 问：原文有保证类短语，卡上却读成不要保证（D3）。F9′ 注明或问：原文的速度设施词没进机制（带否定、当数据用的注明，
  其余问）。F9″ 问：用户不认字段读法，原来会拒的降成问。
- A9 问：点名的方法映射不上可测方案（原因码和条款号同名）。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Iterable

from scipy.stats import norm

from .gate import e_max
from .records import append_jsonl

RULES_VERSION = "intake-v007.1｜验收标准 v007 第八节"

REJECT, ASK, NOTE = "拒", "问", "转译注明"          # 命中的动作
DISPOSITIONS = ("拒", "问", "转译")
TERMINAL_STATES = ("拒", "已转译", "等待回复", "挂起", "用户覆盖")   # 拒单日志的终态
USER, INFERRED = "用户原文", "顶层推断"

# 需求卡的字段（D1）：市场、目标收益、风险、观点、尾部情景、点名的方法与特征等
FIELD_NAMES = ("raw", "r_target", "zero_risk", "floor", "sigma_max", "mdd_max", "risk_lbl", "obj_lbl",
               "tau", "U", "inst", "mech", "feats", "m_pref", "v", "tail")
LIST_FIELDS = ("U", "inst", "mech", "feats", "m_pref")

F1_REASON = "本系统只凭有限考试交付预测模型，交付不了确定性"     # D4 的固定理由，逐字
MIN_REACTION = "约 1 个交易日（默认只提示不下单，每日检查才执行）"


# ---------------------------------------------------------------------------
# 需求卡
# ---------------------------------------------------------------------------
@dataclass
class RequirementCard:
    """需求卡（D1）：固定字段 + 每字段的来源与原文出处。

    sources 的键：标量字段用字段名；列表字段每一项用「字段:值」（如 'U:石油'）。值是 (来源, 原文出处)，
    来源 ∈ {用户原文, 顶层推断}。没登记来源的按「用户原文」读（用户点名的东西要问，不替他换）。
    """
    raw: str
    r_target: float | None = None   # 目标年化收益（小数）
    zero_risk: int = 0              # 要求收益确定（零风险）∈ {0, 1}
    floor: int = 0                  # 要求保本 ∈ {0, 1}；只在原文有明确保证类用词时取 1
    sigma_max: float | None = None  # 可承受的年化波动上限
    mdd_max: float | None = None    # 可承受的最大回撤
    risk_lbl: str | None = None     # 风险偏好 ∈ {低, 中, 高}
    obj_lbl: str | None = None      # 收益取向 ∈ {保值, 稳健, 激进}
    tau: str | None = None          # 持仓 / 决策周期（日内、日频……）
    U: list[str] = field(default_factory=list)       # 市场与价格敞口（如「A股」「石油」）
    inst: list[str] = field(default_factory=list)    # 交易工具类别
    mech: list[str] = field(default_factory=list)    # 赚钱机制（只从用户原话抽）
    feats: list[str] = field(default_factory=list)   # 点名的特征与数据源
    m_pref: list[str] = field(default_factory=list)  # 点名的方法（A9：方向由人给）
    v: str | None = None            # 观点「相信……」
    tail: str | None = None         # 尾部情景「一旦……」
    sources: dict[str, tuple[str, str]] = field(default_factory=dict)

    def put(self, name: str, value, source: str = USER, quote: str = "") -> None:
        """写一个字段（列表字段则追加一项，去重），同时记来源与原文出处。"""
        if name not in FIELD_NAMES or name == "raw":
            raise ValueError(f"需求卡没有字段 {name!r}")
        if name in LIST_FIELDS:
            items = getattr(self, name)
            if value not in items:
                items.append(value)
            self.sources[f"{name}:{value}"] = (source, quote)
        else:
            setattr(self, name, value)
            self.sources[name] = (source, quote)

    def source_of(self, name: str, item=None) -> str:
        return self.sources.get(name if item is None else f"{name}:{item}", (USER, ""))[0]

    def quote_of(self, name: str, item=None) -> str:
        return self.sources.get(name if item is None else f"{name}:{item}", (USER, ""))[1]

    def echo(self) -> list[str]:
        """「我读到的需求」：字段回显 + 来源 + 原文出处（D1）。"""
        lines = []
        for name in FIELD_NAMES[1:]:
            value = getattr(self, name)
            if name in LIST_FIELDS:
                for item in value:
                    lines.append(f"{name} = {item}（{self.source_of(name, item)}：「{self.quote_of(name, item)}」）")
            elif value is not None and (value != 0 or name in ("zero_risk", "floor")):
                shown = _pct(value) if name in ("r_target", "sigma_max", "mdd_max") else value
                src = f"（{self.source_of(name)}：「{self.quote_of(name)}」）" if name in self.sources else ""
                lines.append(f"{name} = {shown}{src}")
        return lines


def _pct(x: float) -> str:
    return f"{x * 100:.4g}%"


# ---------------------------------------------------------------------------
# 收单边界（第九节）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Boundary:
    """收单边界：数据登记表（价格源、特征、代理、利率观测值）、价格敞口表、衍生品表、方法表、速度设施表。

    带哈希进拒单日志（同卡 + 同边界 ⇒ 同处置）；需求触到时再定，定下后照实际更新、照记，不算登记被改。默认值照第九节：
    - 市场表只登实际接通的价格源，默认只登 A 股（作者自用的流水线目前接的是 Qlib 社区的 A 股数据）；特征表登它原样有的
      量价字段和由它们算的量；
    - 代理表空着，要换代理时才逐条登记 (名字, 代理, 版本, 承载度说明)；
    - r_cert 不用原型的示例数：两样利率观测值没登记齐就是 None，要保证的需求只问不拒。
    """
    markets: tuple[str, ...] = ("A股",)
    available_features: tuple[str, ...] = ("开盘价", "最高价", "最低价", "收盘价", "均价", "成交量", "成交额",
                                           "涨跌幅", "收益率", "波动率")
    feature_proxies: tuple[tuple[str, str, str, str], ...] = ()     # (特征, 代理, 版本, 承载度说明)
    exposures: tuple[str, ...] = ("石油", "黄金", "天然气")          # 价格敞口：默认实现会落到衍生品
    exposure_proxies: tuple[tuple[str, str, str, str], ...] = ()    # (敞口, 代理, 版本, 承载度说明)
    derivatives: tuple[str, ...] = ("期货", "期权", "互换", "差价合约", "权证")
    known_methods: tuple[str, ...] = ("传统计量", "计量", "稀疏模型", "稀疏", "LASSO", "岭回归", "线性回归",
                                      "因子模型", "神经网络", "深度学习", "GBDT", "LightGBM", "XGBoost",
                                      "随机森林", "机器学习")
    # D5 只认条款原文那三类：亚秒级反应（含毫秒级、微秒级）、托管机房、逐笔订单簿报撤单。「高频」「逐笔」「延迟套利」
    # 单独出现不在这里：没说非要这三样设施，只问（F2″）
    excluded_facilities: tuple[str, ...] = ("亚秒级", "毫秒级", "微秒级", "托管机房", "逐笔报撤单")
    slow_analogs: tuple[tuple[str, str], ...] = (
        ("做市", "日频短期反转"),
        ("提供流动性", "日频短期反转"),
        ("抢第一波", "事件后的延续 / 反转（事件发生 ≥ 1 个交易日之后那段）"),
    )
    r_tbill: float | None = None     # 同币种短期国债收益率（观测值）
    r_deposit: float | None = None   # 政府担保存款利率（观测值）
    r_f: float | None = None         # 无风险利率：只用于展示目标折合的 Sharpe，不进任何拒（D4）
    n_fin: int = 5                   # 送检数上限（第九节默认）：只用于 D10 展示
    t_val_days: int = 252            # T_val（第九节默认 [GUESS]）：只用于 D10 展示
    k_clarify: int = 3               # 反问轮数上限 [GUESS]：到了就挂起，不拒（D8）

    @property
    def r_cert(self) -> float | None:
        """r_cert = 同币种短期国债收益率与政府担保存款利率的较大者（D4）；两样没登记齐就是 None。"""
        if self.r_tbill is None or self.r_deposit is None:
            return None
        return max(self.r_tbill, self.r_deposit)

    def hash(self) -> str:
        """边界哈希（进拒单日志）。"""
        blob = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# 示例边界：演示与回归用例用，不是登记值（第九节：正式跑时 r_cert 不用这些示例数，市场表只登实际接通的价格源）
EXAMPLE_BOUNDARY = Boundary(
    markets=("美股", "A股", "港股", "日股"),
    available_features=("成交量", "换手率", "市盈率", "收益率", "波动率"),
    feature_proxies=(
        ("伊朗局势", "Caldara–Iacoviello 的地缘政治风险指数 GPR", "示例", "承载度未核"),
        ("局势动荡", "Caldara–Iacoviello 的地缘政治风险指数 GPR", "示例", "承载度未核"),
        ("地缘政治风险", "Caldara–Iacoviello 的地缘政治风险指数 GPR", "示例", "承载度未核"),
        ("Truth Social 发帖", "Truth Social 历史帖子的日频汇编（按 ≥ 1 个交易日的时段汇总）", "示例", "承载度未核"),
    ),
    exposure_proxies=(
        ("石油", "能源股、持股型能源基金", "示例", "承载度未核"),
        ("黄金", "金矿股、实物黄金 ETF", "示例", "承载度未核"),
        ("天然气", "能源股", "示例", "承载度未核"),
    ),
    r_tbill=0.04, r_deposit=0.03, r_f=0.04,
)


# ---------------------------------------------------------------------------
# 关键词替身：原文 → 需求卡
# ---------------------------------------------------------------------------
_PCT = r"(\d+(?:\.\d+)?)\s*[%％]"
_R_TARGET_RE = re.compile(r"年化(?:收益(?:率)?)?\s*(?:要|达到|做到)?\s*" + _PCT)
_SIGMA_RE = re.compile(r"波动(?:率)?\s*(?:不超过|小于|控制在|上限)?\s*" + _PCT)
_MDD_RE = re.compile(r"(?:最大)?回撤\s*(?:不超过|小于|控制在|上限)?\s*" + _PCT)
_DOUBLE_RE = re.compile(r"翻倍|翻番")
# 保证类短语（D1：零风险、风险为 0、无风险、稳赚、保证收益、保本）；「无风险利率」不算
_GUARANTEE_RE = re.compile(r"零风险|风险\s*为\s*[0０零]|无风险(?!利率|收益率)|稳赚|保证收益|保本")
_ZERO_RISK_RE = re.compile(r"零风险|风险\s*为\s*[0０零]|无风险(?!利率|收益率)|稳赚|保证收益")
_FLOOR_RE = re.compile(r"保本|保证不亏|不亏本金")
_NEGATIONS = ("没有", "不是", "不要", "不存在", "并非", "不可能", "不求", "无需", "不追求", "不需要", "不用", "不指望",
              "不做", "不碰", "不想要")
_CLAUSE_END = "，,。；;！!？?\n"      # 否定词只在同一分句里往前找（「不做日内，做美股」里的美股不算被否定）
# 跟在短语后面的否定（「零风险是不可能的」「美股不做」「托管机房不要」）
_NEG_AFTER_RE = re.compile(r"\s*(?:是|也是|根本|就)?\s*(?:不可能|不现实|不存在|做不到|不做|不碰|不要|除外)")
# 跟在工具 / 设施词后面、说明是拿它当数据用的字（「期货持仓数据」「逐笔数据」「逐笔成交」「高频因子」）
_FEATURE_CONTEXT_RE = re.compile(r"\s*(?:数据|持仓|指标|价格|行情|因子|特征|成交)")

_RISK_WORDS = {"高风险": "高", "中风险": "中", "中等风险": "中", "低风险": "低", "保守": "低"}
_OBJ_WORDS = {"高收益": "激进", "翻倍": "激进", "翻番": "激进", "暴利": "激进", "激进": "激进",
              "稳健": "稳健", "稳一点": "稳健", "保值": "保值"}
_TAU_WORDS = {"日内": "日内", "日频": "日频", "周频": "周频", "月频": "月频", "月度": "月频", "长期": "长期"}
_MARKET_WORDS = {"美股": "美股", "美国股市": "美股", "标普": "美股", "纳斯达克": "美股", "A股": "A股",
                 "A 股": "A股", "沪深": "A股", "港股": "港股", "恒生": "港股", "日股": "日股", "日经": "日股",
                 "朝鲜": "朝鲜股市"}
_EXPOSURE_WORDS = {"原油": "石油", "石油": "石油", "油价": "石油", "黄金": "黄金", "金价": "黄金", "天然气": "天然气"}
_INFERRED_EXPOSURE = {"伊朗": "石油"}   # 顶层推断的示例（初始计划第 5 节的转译示例：伊朗 → 石油）
_INST_WORDS = {"期货": "期货", "期权": "期权", "互换": "互换", "差价合约": "差价合约", "CFD": "差价合约", "权证": "权证"}
_FACILITY_WORDS = {"高频": "高频", "亚秒": "亚秒级", "毫秒": "毫秒级", "微秒": "微秒级", "托管机房": "托管机房",
                   "co-location": "托管机房", "逐笔": "逐笔", "报撤单": "逐笔报撤单", "延迟套利": "延迟套利"}
_MECH_WORDS = {**_FACILITY_WORDS, "做市": "做市", "提供流动性": "提供流动性", "抢第一波": "抢第一波",
               "抢新闻": "抢第一波", "短期反转": "短期反转", "动量": "动量", "均值回归": "均值回归",
               "事件驱动": "事件驱动"}
_FEAT_WORDS = {"披萨指数": "五角大楼披萨指数", "Truth Social": "Truth Social 发帖", "伊朗局势": "伊朗局势",
               "局势动荡": "局势动荡", "地缘政治": "地缘政治风险", "成交量": "成交量", "换手率": "换手率",
               "市盈率": "市盈率"}
_METHOD_WORDS = ("传统计量", "计量", "稀疏模型", "稀疏", "LASSO", "岭回归", "线性回归", "因子模型", "神经网络",
                 "深度学习", "GBDT", "LightGBM", "XGBoost", "随机森林", "机器学习", "辩证法", "易经", "占星",
                 "风水", "缠论", "波浪理论")
_METHOD_FALLBACK_RE = re.compile(r"用([一-鿿A-Za-z]{1,6}?)(?:给我|帮我)做")
_VIEW_RE = re.compile(r"(?:相信|看好|认为)([^。；;，,]+)")
_TAIL_RE = re.compile(r"一旦([^，,。；;]+?)(?:发生)?[，,]\s*(?:就|会)?有?尾部风险")


def _find_terms(text: str, words) -> list[tuple[int, int, str, str]]:
    """关键词查找：长词优先、不重叠；返回 [(起, 止, 原文片段, 映射值)]，按位置排序。"""
    mapping = words if isinstance(words, dict) else {w: w for w in words}
    taken: list[tuple[int, int]] = []
    found = []
    for w in sorted(mapping, key=len, reverse=True):
        for m in re.finditer(re.escape(w), text, flags=re.IGNORECASE):
            s, e = m.span()
            if any(s < te and ts < e for ts, te in taken):
                continue
            taken.append((s, e))
            found.append((s, e, m.group(0), mapping[w]))
    return sorted(found)


def _negated(text: str, start: int, end: int | None = None) -> bool:
    """带否定：短语前 6 个字里（只看同一分句）有否定词（「我知道没有零风险的策略」「不要托管机房」「不做美股」），
    或短语后面紧跟「是不可能的 / 不做 / 除外」这类话（「零风险是不可能的」「美股不做」）。"""
    head = text[max(0, start - 6):start]
    head = head[max(head.rfind(c) for c in _CLAUSE_END) + 1:]
    if any(neg in head for neg in _NEGATIONS):
        return True
    return end is not None and _NEG_AFTER_RE.match(text, end) is not None


def _feature_context(text: str, end: int) -> bool:
    return _FEATURE_CONTEXT_RE.match(text, end) is not None


def extract_card(raw: str) -> RequirementCard:
    """关键词替身：把原文抽成需求卡（D1；顶层接上后由顶层抽）。

    mech 只从用户原话抽；floor 只在明确保证类用词时取 1，「尽量少亏」不算；带否定的保证词（「没有零风险」「零风险是
    不可能的」）不置 1，交给 D3 的短语核对去问。带否定的市场（「不做美股」）不进 U，带否定的机制（「不要高频」）不进 mech。
    """
    card = RequirementCard(raw=raw)
    t = raw
    m = _R_TARGET_RE.search(t)
    if m:
        card.put("r_target", float(m.group(1)) / 100, USER, m.group(0))
    elif _DOUBLE_RE.search(t):
        card.put("r_target", 1.0, INFERRED, _DOUBLE_RE.search(t).group(0))
    for regex, name in ((_SIGMA_RE, "sigma_max"), (_MDD_RE, "mdd_max")):
        m = regex.search(t)
        if m:
            card.put(name, float(m.group(1)) / 100, USER, m.group(0))
    for regex, name in ((_ZERO_RISK_RE, "zero_risk"), (_FLOOR_RE, "floor")):
        for m in regex.finditer(t):
            if not _negated(t, m.start(), m.end()):
                card.put(name, 1, USER, m.group(0))
                break
    for words, name in ((_RISK_WORDS, "risk_lbl"), (_OBJ_WORDS, "obj_lbl"), (_TAU_WORDS, "tau")):
        terms = _find_terms(t, words)
        if terms:
            card.put(name, terms[0][3], USER, terms[0][2])
    for s, e, quote, value in _find_terms(t, _MARKET_WORDS) + _find_terms(t, _EXPOSURE_WORDS):
        if not _negated(t, s, e):                        # 「不做美股」「美股除外」不算要做的市场
            card.put("U", value, USER, quote)
    for _, _, quote, value in _find_terms(t, _INFERRED_EXPOSURE):
        if value not in card.U:
            card.put("U", value, INFERRED, quote)
    for _, e, quote, value in _find_terms(t, _INST_WORDS):
        if not _feature_context(t, e):
            card.put("inst", value, USER, quote)
    for s, e, quote, value in _find_terms(t, _MECH_WORDS):
        if _negated(t, s, e) or (value in _FACILITY_WORDS.values() and _feature_context(t, e)):
            continue                                     # 带否定的、当数据用的（「逐笔数据」）不进机制，D3 在回显里注明
        card.put("mech", value, USER, quote)
    for _, _, quote, value in _find_terms(t, _FEAT_WORDS):
        card.put("feats", value, USER, quote)
    for _, _, quote, value in _find_terms(t, _METHOD_WORDS):
        card.put("m_pref", value, USER, quote)
    m = _METHOD_FALLBACK_RE.search(t)
    if m and not card.m_pref:
        card.put("m_pref", m.group(1), USER, m.group(0))
    for regex, name in ((_VIEW_RE, "v"), (_TAIL_RE, "tail")):
        m = regex.search(t)
        if m:
            card.put(name, m.group(1).strip(), USER, m.group(0))
    return card


# ---------------------------------------------------------------------------
# 规则：按条款号 D3 → D8、A9
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Hit:
    """命中表一行：条款号、原因码、动作、触发字段与值、给用户的话、可回的选项。"""
    rule: str
    code: str
    action: str                       # 拒 / 问 / 转译注明
    field: str
    value: str
    message: str
    options: tuple[str, ...] = ()
    waived: bool = False              # 用户回「照做」解除（只对「问」有效）


@dataclass(frozen=True)
class Assessment:
    """一次收单的输出（D1）。"""
    disposition: str                  # 拒 / 问 / 转译
    hits: tuple[Hit, ...]
    message: str
    echo: tuple[str, ...]             # 「我读到的需求」
    notes: tuple[str, ...]            # 转译注明
    header: tuple[str, ...]           # 方案抬头（D9 / D10）：只给人看，不进汇总表
    rules_version: str
    boundary_hash: str

    @property
    def codes(self) -> tuple[str, ...]:
        """没被解除的命中原因码。"""
        return tuple(h.code for h in self.hits if not h.waived)


def _proxy_text(entry) -> str:
    if entry is None:
        return "代理还没登记（要换代理时逐条登记代理、版本与承载度说明）"
    proxy, version, note = entry
    return f"{proxy}（版本 {version}；{note}）"


def _d3_phrases(card: RequirementCard, b: Boundary) -> list[Hit]:
    """D3：原文的保证类短语、设施类短语和卡上字段对不上——会导致拒的改成问，只影响转译的在回显里注明。"""
    hits = []
    phrases = [m.group(0) for m in _GUARANTEE_RE.finditer(card.raw)]
    if phrases and not card.zero_risk and not card.floor:
        hits.append(Hit("D3", "F9", ASK, "zero_risk / floor", "0 / 0",
                        f"原文出现「{phrases[0]}」，可需求卡读成不要求保证（zero_risk = 0、floor = 0）。"
                        "请确认：你要不要收益确定或保本？", ("不要保证，照需求卡做", "要保证")))
    for start, end, quote, value in _find_terms(card.raw, _FACILITY_WORDS):
        if value in card.mech:
            continue
        if _negated(card.raw, start, end):
            hits.append(Hit("D3", "F9′", NOTE, "mech", quote, f"原文「{quote}」带否定（不要 / 没有……），按不要读，不进机制。"))
        elif _feature_context(card.raw, end):
            hits.append(Hit("D3", "F9′", NOTE, "mech", quote, f"原文出现「{quote}」，按特征 / 数据读，不按速度设施读。"))
        else:
            hits.append(Hit("D3", "F9′", ASK, "mech", quote,
                            f"原文出现「{quote}」，可需求卡的机制里没有它。请确认你是不是要用它。",
                            ("是，要用", "不是，照需求卡做")))
    return hits


def _d4_guarantee(card: RequirementCard, b: Boundary) -> list[Hit]:
    """D4：要零风险或保本、且目标收益高于 r_cert → 拒（F1）；目标不高于 r_cert、没写，或 r_cert 没登记 → 问（F1′）。"""
    if not (card.zero_risk or card.floor):
        return []
    name = "zero_risk" if card.zero_risk else "floor"
    quote = card.quote_of(name) or ("零风险" if card.zero_risk else "保本")
    r_cert = b.r_cert
    if card.r_target is not None and r_cert is not None and card.r_target > r_cert:
        target = card.quote_of("r_target") or _pct(card.r_target)
        return [Hit("D4", "F1", REJECT, name, f"{name} = 1；r_target = {_pct(card.r_target)}",
                    f"拒绝（F1）。你要「{quote}」（{name} = 1），还要「{target}」，高过 r_cert = {_pct(r_cert)}"
                    f"（同币种短期国债与政府担保存款利率的较大者）。这是要保证：{F1_REASON}。"
                    "如果字段读错了，直接回复更正。")]
    if card.r_target is None:
        why = "没写目标收益"
    elif r_cert is None:
        why = f"目标收益 {_pct(card.r_target)}，可数据登记表里还没有 r_cert 的观测值，比不出高低"
    else:
        why = f"目标收益 {_pct(card.r_target)} 不高于 r_cert = {_pct(r_cert)}，存款或国债就够了，用不着模型"
    return [Hit("D4", "F1′", ASK, name, f"{name} = 1", f"你要「{quote}」，{why}。愿不愿意接受一定风险？",
                ("接受一定风险，改需求", "撤回"))]


def _d5_speed(card: RequirementCard, b: Boundary) -> list[Hit]:
    """D5：明说要亚秒级反应、托管机房、逐笔订单簿报撤单才能做 → 拒（F2）；「高频」「逐笔」「延迟套利」这类词单独出现、
    没说非要这三样设施 → 问（F2″：真高频做不了，按日频做行不行）；有不靠这些设施的同类机制就转译（F2′）。"""
    facilities = [x for x in card.mech if x in b.excluded_facilities]
    vague = [x for x in card.mech if x in _FACILITY_WORDS.values() and x not in b.excluded_facilities]
    slow = dict(b.slow_analogs)
    analogs = [(x, slow[x]) for x in card.mech if x in slow]
    if facilities:
        quotes = "、".join(card.quote_of("mech", x) or x for x in facilities)
        msg = (f"拒绝（F2）。需求明说要「{quotes}」才能做——亚秒级反应、托管机房、逐笔订单簿报撤单这类设施本系统没有，"
               f"最短反应时间{MIN_REACTION}。")
        if analogs:
            msg += ("不靠这些设施的同类可以做：" + "；".join(f"「{x}」→「{y}」" for x, y in analogs)
                    + "，慢版本的利润有过明显衰减、对成本敏感。")
        return [Hit("D5", "F2", REJECT, "mech", "、".join(facilities), msg + "如果字段读错了，直接回复更正。")]
    hits = [Hit("D5", "F2″", ASK, "mech", x,
                f"你提到「{card.quote_of('mech', x) or x}」。本系统没有亚秒级反应、托管机房、逐笔订单簿报撤单这类设施，"
                f"最短反应时间{MIN_REACTION}：拿这类数据做日频特征、或做它的慢版本可以，非要这么快才能做的做不了。"
                "按日频做行不行？", ("按日频做", "非要这么快（那就做不了）")) for x in vague]
    return hits + [Hit("D5", "F2′", NOTE, "mech", x,
                       f"机制「{x}」换成不靠速度设施的同类「{y}」；方案注明慢版本的利润有过明显衰减、对成本敏感。")
                   for x, y in analogs]


def _d6_derivatives(card: RequirementCard, b: Boundary) -> list[Hit]:
    """D6：用户点名衍生品只问不拒（F3）；顶层推断的落到衍生品就换非衍生品候选（F3′）；用户点名价格敞口却只能用代理，也问（F3″）。"""
    hits = []
    for x in card.inst:
        if x not in b.derivatives:
            continue
        if card.source_of("inst", x) == USER:
            hits.append(Hit("D6", "F3", ASK, "inst", x,
                            f"你点名了衍生品「{x}」。衍生品只问不拒（U-2 只暂封衍生品定价研究）：改成非衍生品行不行？",
                            ("改成非衍生品", "照做", "撤回")))
        else:
            hits.append(Hit("D6", "F3′", NOTE, "inst", x,
                            f"顶层推断的工具「{x}」是衍生品，换成非衍生品候选；代理承载度（代理能替原敞口多少）待检验。"))
    proxies = {e: (p, ver, note) for e, p, ver, note in b.exposure_proxies}
    for u in card.U:
        if u not in b.exposures:
            continue
        alt = _proxy_text(proxies.get(u))
        if card.source_of("U", u) == USER:
            hits.append(Hit("D6", "F3″", ASK, "U", u,
                            f"你点名的价格敞口「{u}」只能用代理实现：{alt}；代理能替原敞口多少待检验。接受代理吗？",
                            ("接受代理", "撤回")))
        else:
            hits.append(Hit("D6", "F3′", NOTE, "U", u,
                            f"顶层推断的敞口「{u}」（原文「{card.quote_of('U', u)}」）默认会落到衍生品，"
                            f"换成非衍生品候选：{alt}；代理承载度待检验。"))
    return hits


def _d7_data(card: RequirementCard, b: Boundary) -> list[Hit]:
    """D7：卡上的市场在数据登记表里一个价格源都没有 → 拒（F4）；有的接通、有的没接通 → 只做接通的，没接通的在回显里
    注明（F4″，不拒：顺带提到的市场多半只是参考）；点名特征拿不到，有代理就换、没有就删（F4′）；点名的全删光 → 问（F8）。"""
    hits = []
    named = [u for u in card.U if u not in b.exposures]          # 价格敞口（石油等）归 D6 管
    connected = [u for u in named if u in b.markets]
    for u in named:
        if u in b.markets:
            continue
        if connected:
            hits.append(Hit("D7", "F4″", NOTE, "U", u,
                            f"市场「{u}」在数据登记表里没有价格源：这次只做已接通的"
                            f"{'、'.join(f'「{x}」' for x in connected)}；要做「{u}」得先接上它的价格源"
                            "（只是拿它当参考的话，不用管这条）。"))
            continue
        hits.append(Hit("D7", "F4", REJECT, "U", u,
                        f"拒绝（F4）。你点名的市场「{u}」在数据登记表里没有任何价格源，没法训练也没法考试。"
                        f"登记表里有：{'、'.join(b.markets)}。换一个有数据的市场再提；如果字段读错了，直接回复更正。"))
    proxies = {f: (p, ver, note) for f, p, ver, note in b.feature_proxies}
    deleted = []
    for f in card.feats:
        if f in b.available_features:
            continue
        if f in proxies:
            hits.append(Hit("D7", "F4′", NOTE, "feats", f, f"特征「{f}」拿不到原样数据，换成代理：{_proxy_text(proxies[f])}。"))
        else:
            hits.append(Hit("D7", "F4′", NOTE, "feats", f, f"特征「{f}」拿不到、也没有代理，删去。"))
            deleted.append(f)
    named = [f for f in card.feats if card.source_of("feats", f) == USER]
    if named and all(f in deleted for f in named):
        hits.append(Hit("D7", "F8", ASK, "feats", "、".join(named),
                        "你点名的特征全都拿不到，剩下的只有通用特征，和你的想法无关，还做吗？",
                        ("照做（只用通用特征）", "换特征", "撤回")))
    return hits


def _d8_risk(card: RequirementCard, b: Boundary) -> list[Hit]:
    """D8：要低风险或保本、又要激进 → 问选一边（F6）；有目标收益而波动、回撤都空 → 照转译并标注（F7 / F7′）。
    不设「问 k 次就拒」：反问轮数到上限就挂起（见 make_log_entry）。"""
    hits = []
    if (card.risk_lbl == "低" or card.floor) and card.obj_lbl == "激进":
        left = card.quote_of("risk_lbl") or card.quote_of("floor") or "低风险"
        right = card.quote_of("obj_lbl") or "激进"
        hits.append(Hit("D8", "F6", ASK, "risk_lbl / obj_lbl", f"{card.risk_lbl or 'floor = 1'} / 激进",
                        f"你说「{left}」，又说「{right}」，两边打架：选一边，或撤回一句。",
                        ("改风险偏好", "改收益目标", "撤回")))
    if card.r_target is not None and card.sigma_max is None and card.mdd_max is None:
        if card.risk_lbl is not None or card.floor:
            hits.append(Hit("D8", "F7", NOTE, "sigma_max / mdd_max", "空",
                            f"给了目标收益 {_pct(card.r_target)}，没给波动或回撤上限：按风险偏好"
                            f"「{card.risk_lbl or '保本'}」转译，方案里标注，使用前对照。"))
        elif not card.zero_risk:
            hits.append(Hit("D8", "F7′", NOTE, "风险信息", "全空",
                            f"只给了目标收益 {_pct(card.r_target)}、没给任何风险信息：照转译，方案里注明隐含的最低波动——"
                            f"粗算波动 ≈ 目标收益 ÷ Sharpe，Sharpe 为 1 时约 {_pct(card.r_target)}、为 2 时约 "
                            f"{_pct(card.r_target / 2)}（只作提示）。"))
    return hits


def _a9_method(card: RequirementCard, b: Boundary) -> list[Hit]:
    """A9：方向由人给——点名的方法映射不上可测方案就问。"""
    unmapped = [x for x in card.m_pref if x not in b.known_methods]
    if not unmapped:
        return []
    names = "、".join(unmapped)
    return [Hit("A9", "A9", ASK, "m_pref", names,
                f"你指定的方法「{names}」映射不到可测的方案。方向由人给，映射不上就问：能说具体一点吗——"
                "一篇论文、一个模型名；或者回「照做」，让顶层自己提方案。", ("补充方法", "照做（顶层自己提）", "撤回"))]


RULES = (_d3_phrases, _d4_guarantee, _d5_speed, _d6_derivatives, _d7_data, _d8_risk, _a9_method)


def _header(card: RequirementCard, b: Boundary) -> list[str]:
    """方案抬头（只给人看，不进汇总表）：D10 纯噪音能碰到的成绩（E_max 按 n_fin 算）与越过目标的概率；D9 衡量口径。"""
    years = b.t_val_days / 252
    lines = [f"按你的送检上限和验证期，纯噪音能碰到的成绩约 {e_max(b.n_fin) / math.sqrt(years):.2f}"
             f"（年化 Sharpe；E_max(n_fin)·√(252/T_val)，n_fin = {b.n_fin}，T_val = {b.t_val_days} 个交易日）。"]
    if card.r_target is not None and card.sigma_max:
        r_f = 0.0 if b.r_f is None else b.r_f
        sr_req = (card.r_target - r_f) / card.sigma_max
        p_cross = 1.0 - norm.cdf(sr_req * math.sqrt(years)) ** b.n_fin
        lines.append(f"你的目标折合年化 Sharpe ≈ {sr_req:.2f}（r_f = {_pct(r_f)}"
                     f"{'' if b.r_f is not None else '，未登记按 0 算'}）；纯噪音越过它的概率 ≈ {p_cross:.2f}（只展示）。")
    basis = "、".join(x for x in (f"观点「{card.v}」" if card.v else "", f"尾部情景「{card.tail}」" if card.tail else "")
                      if x)
    lines.append(f"给你看的指标从{basis}来（D9）。" if basis else "需求没说衡量口径，给你看的指标默认用年化 Sharpe（D9）。")
    return lines


def _plain_notes(card: RequirementCard) -> list[str]:
    notes = []
    if card.tau == "日内":
        notes.append("周期「日内」照留：默认只提示、每日检查才执行，实际等于日频；要日内落地得开自动执行。"
                     "日内本身不触发任何规则（D12）。")
    if card.v is None or card.tail is None:
        notes.append("观点 v / 尾部情景 tail 缺：由顶层补、标「推断」，写进方案供使用前对照（D8）。")
    return notes


def _fields_of(h: Hit) -> set[str]:
    return {f.strip() for f in h.field.split("/")}


def _disputed_ask(h: Hit) -> Hit:
    """D1–D3：用户不认字段读法就不拒——这条「拒」降成「问」（F9″），回显原文出处、请用户改写。"""
    return Hit("D3", "F9″", ASK, h.field, h.value,
               f"你不认「{h.field}」这个读法，先不拒：请把原文里对应的那句改写一下（我读到的出处见回显），"
               f"我按新写法重新抽卡。原来会拒的理由：{h.message}", ("改写这一句", "撤回"))


def assess(card: RequirementCard, boundary: Boundary | None = None, waived: Iterable[str] = (),
           disputed: Iterable[str] = ()) -> Assessment:
    """按卡确定地算出全部命中与处置（D1）。

    处置：有「拒」命中 → 拒（只可能来自 F1 / F2 / F4）；否则有没解除的「问」→ 问；否则转译。
    waived 是用户已回「照做」的原因码（只解除「问」，解除不了「拒」；D12：问过后用户说照做不是拒的理由）。
    disputed 是用户不认读法的字段名（如 "zero_risk"、"floor"、"mech"、"U"）：这些字段触发的「拒」降成「问」（F9″），
    回显原文出处、请用户改写（D1–D3：用户不认字段读法就不拒）。
    """
    b = Boundary() if boundary is None else boundary
    waived, disputed = set(waived), set(disputed)
    hits = [h for rule in RULES for h in rule(card, b)]
    hits = [_disputed_ask(h) if h.action == REJECT and disputed & _fields_of(h) else h for h in hits]
    hits = [replace(h, waived=(h.action == ASK and h.code in waived)) for h in hits]
    live = [h for h in hits if not h.waived]
    if any(h.action == REJECT for h in live):
        disposition = "拒"
        message = "\n".join(h.message for h in live if h.action == REJECT)
    elif any(h.action == ASK for h in live):
        disposition = "问"
        message = "需要你先回答：\n" + "\n".join(
            f"[{h.code}] {h.message}（可回：{' / '.join(h.options) or '照做'}）" for h in live if h.action == ASK)
    else:
        disposition = "转译"
        done = [h.message for h in live if h.action == NOTE]
        message = "转译。" + ("".join(done) if done else "零替换，照原需求落成研究方案。") + "".join(_plain_notes(card))
    notes = [h.message for h in live if h.action == NOTE] + _plain_notes(card)
    header = _header(card, b) if disposition == "转译" else []
    return Assessment(disposition, tuple(hits), message, tuple(card.echo()), tuple(notes), tuple(header),
                      RULES_VERSION, b.hash())


# ---------------------------------------------------------------------------
# 拒单日志（只追加）
# ---------------------------------------------------------------------------
def make_log_entry(req_id: str, card: RequirementCard, result: Assessment, *, clarify_rounds: int = 0,
                   boundary: Boundary | None = None, now: datetime | None = None) -> dict:
    """拒单日志一行：时间、需求 ID、原文、需求卡（字段、来源、出处）、规则集版本与边界哈希、命中表、处置、消息、终态。

    终态：拒 / 已转译 / 等待回复；反问轮数到上限 → 挂起（不拒，D8）；问过后用户回「照做」→ 用户覆盖。
    """
    b = Boundary() if boundary is None else boundary
    terminal = {"拒": "拒", "问": "等待回复", "转译": "已转译"}[result.disposition]
    if result.disposition == "问" and clarify_rounds >= b.k_clarify:
        terminal = "挂起"
    if result.disposition == "转译" and any(h.waived for h in result.hits):
        terminal = "用户覆盖"
    card_dict = {name: getattr(card, name) for name in FIELD_NAMES}
    card_dict["sources"] = {k: list(v) for k, v in sorted(card.sources.items())}
    return {
        "time": (now or datetime.now()).isoformat(timespec="seconds"),
        "req_id": req_id,
        "raw": card.raw,
        "card": card_dict,
        "rules_version": result.rules_version,
        "boundary_hash": result.boundary_hash,
        "hits": [asdict(h) for h in result.hits],
        "disposition": result.disposition,
        "message": result.message,
        "terminal": terminal,
    }


def append_log(path, entry: dict) -> None:
    """拒单日志只追加。"""
    if entry.get("terminal") not in TERMINAL_STATES:
        raise ValueError(f"终态只能是 {TERMINAL_STATES}")
    append_jsonl(path, entry)
