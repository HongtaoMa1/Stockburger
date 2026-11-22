# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""审计账：每次调用顶层记一行（v007 第七节 C1 / C2）。

一句话：顶层大模型的知识截止日可能盖住测试期（它「记得」那段行情）。这类前视泄漏不设硬闸门（U-3），只记下来作参考：
每次调用顶层，脚本往审计账写一行——模型 ID、调用日期、厂商申报的截止月 t_c（取月末）、测试段 [t_1, t_2]、重叠月数
m_ov、按日期算的截止日标签 L_label，以及这次提示用了哪种禁令句、句子哈希。

规格：验收标准 v007 第七节 C1 / C2。
- L_label：t_c < t_1 为 clean，t_1 ≤ t_c < t_2 为 partial，t_c ≥ t_2 为 contaminated（和阶梯层数 L 是两回事）。
- L_label ≠ clean 时，日报里测试段成绩旁标「压力测试 / 概念验证级」——按 TEST_SCORE_TAG 查表印，不算分支。
- C2（硬）：代码不许拿 L_label、m_ov 驱动闸门、挑候选或改流程。static_branch_check 是可选的静态搜索验法（U-5）：
  只看语法树里的分支条件，拿字典查表代替 if 这类写法搜不出来。
审计账不进顶层汇总表，只进日报（A7 / C6）。
"""
from __future__ import annotations

import ast
import calendar
import json
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from .toplayer_prompt import PromptSpec

# 月数折法：天数 ÷ 格里历平均月长（365.25 / 12 = 30.4375 天）
DAYS_PER_MONTH = 365.25 / 12
L_LABELS = ("clean", "partial", "contaminated")
STRESS_TAG = "压力测试 / 概念验证级"
# 日报里测试段成绩旁的标签（C1）：None = 没调过顶层（顶层还没接上，没有审计行）
TEST_SCORE_TAG = {None: "", "clean": "", "partial": STRESS_TAG, "contaminated": STRESS_TAG}
_BAN_TEXT = {None: "未记", "off": "关", "W1": "开 · W1", "W2": "开 · W2"}
_MONTHS_EN = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}


# ---------------------------------------------------------------------------
# 截止月、重叠月数、截止日标签
# ---------------------------------------------------------------------------
def _as_date(x) -> date:
    """date / datetime / 'YYYY-MM-DD' → date。"""
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x).strip())


def month_end(reported) -> date:
    """厂商公布的截止「月」→ 该月最后一天（C1：t_c 取月末）。

    接受 date / datetime（取其所在月）、'2025-06'、'2025-6'、'2025年6月'、'June 2025'、'Jun 2025'。
    """
    if isinstance(reported, (date, datetime)):
        y, m = reported.year, reported.month
    else:
        s = str(reported).strip()
        hit = re.search(r"(\d{4})\s*[-/.年]\s*(\d{1,2})", s)
        if hit:
            y, m = int(hit.group(1)), int(hit.group(2))
        else:
            hit = re.search(r"([A-Za-z]{3,})\.?\s+(\d{4})", s)
            if not hit or hit.group(1)[:3].lower() not in _MONTHS_EN:
                raise ValueError(f"认不出截止月份：{reported!r}")
            y, m = int(hit.group(2)), _MONTHS_EN[hit.group(1)[:3].lower()]
    return date(y, m, calendar.monthrange(y, m)[1])


def overlap_months(t_c, t_1, t_2) -> float:
    """m_ov = max(0, min(t_c, t_2) − t_1)，单位月（天数 ÷ 30.4375）。"""
    t_c, t_1, t_2 = _as_date(t_c), _as_date(t_1), _as_date(t_2)
    return max(0.0, (min(t_c, t_2) - t_1).days / DAYS_PER_MONTH)


def cutoff_label(t_c, t_1, t_2) -> str:
    """截止日标签 L_label，按日期比较算出、照记（这里的分支只看日期；算出来以后任何代码都不许拿它分支，C2）。"""
    t_c, t_1, t_2 = _as_date(t_c), _as_date(t_1), _as_date(t_2)
    if t_1 > t_2:
        raise ValueError(f"测试段起点晚于终点：t_1 = {t_1}，t_2 = {t_2}")
    if t_c < t_1:
        return "clean"
    if t_c < t_2:
        return "partial"
    return "contaminated"


# ---------------------------------------------------------------------------
# 审计账一行（C1）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CallRecord:
    """审计账一行：每次调用顶层时写（C1）。只当解读参考。"""
    model_id: str              # 模型 ID 原串，不改写
    call_date: date            # 调用日期
    t_c: date                  # 厂商申报的截止月，取月末
    t_1: date                  # 测试段首日
    t_2: date                  # 测试段末日（= 数据末日）
    m_ov: float                # 重叠月数
    L_label: str               # clean / partial / contaminated
    ban_variant: str | None    # 禁令句用了哪种：off / W1 / W2；None = 未记
    ban_hash: str | None       # 所用禁令句的 sha256；关着或未记为 None

    @property
    def test_tag(self) -> str:
        """日报里测试段成绩旁要标的字（C1）：clean 为空，partial / contaminated 为「压力测试 / 概念验证级」。"""
        return TEST_SCORE_TAG[self.L_label]

    def to_json(self) -> dict:
        """审计账 JSON 行（键名照字段名）。"""
        return {"model_id": self.model_id, "call_date": self.call_date.isoformat(), "t_c": self.t_c.isoformat(),
                "t_1": self.t_1.isoformat(), "t_2": self.t_2.isoformat(), "m_ov": round(self.m_ov, 4),
                "L_label": self.L_label, "ban_variant": self.ban_variant, "ban_hash": self.ban_hash}

    def render(self) -> str:
        """人读的几行（进日报，不进汇总表）。"""
        digest = f"；句子哈希 {self.ban_hash[:12]}…" if self.ban_hash else ""
        return "\n".join([
            f"顶层调用审计 · 模型 {self.model_id} · 调用日期 {self.call_date.isoformat()}",
            f"截止月 t_c = {self.t_c.isoformat()}（厂商申报，取月末）；测试段 [t_1, t_2] = "
            f"[{self.t_1.isoformat()}, {self.t_2.isoformat()}]",
            f"m_ov = {self.m_ov:.2f} 月；L_label = {self.L_label}（只当解读参考，不驱动任何流程）",
            f"禁令句 = {_BAN_TEXT[self.ban_variant]}{digest}",
        ])


def make_call_record(model_id: str, call_date, training_cutoff, t_1, t_2, *,
                     prompt: PromptSpec | None = None) -> CallRecord:
    """造一行审计账（C1）。

    training_cutoff 给厂商公布的月份（如 'June 2025'、'2025-06'），取该月末日；prompt 是这次实际发出的提示配置
    （toplayer_prompt.build_schema_only_prompt 返回的 PromptSpec），禁令句写法与句子哈希从它来；没给就记「未记」。
    """
    if prompt is not None and not isinstance(prompt, PromptSpec):
        raise TypeError("prompt 要给 toplayer_prompt.PromptSpec（build_schema_only_prompt 返回的那份）")
    t_c, t_1, t_2 = month_end(training_cutoff), _as_date(t_1), _as_date(t_2)
    return CallRecord(
        model_id=str(model_id), call_date=_as_date(call_date), t_c=t_c, t_1=t_1, t_2=t_2,
        m_ov=overlap_months(t_c, t_1, t_2), L_label=cutoff_label(t_c, t_1, t_2),
        ban_variant=None if prompt is None else prompt.ban_variant,
        ban_hash=None if prompt is None else prompt.ban_hash,
    )


def append_jsonl(path, row: dict) -> None:
    """只追加写一行 JSON（审计账、拒单日志共用）；不改、不删已有行。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


# ---------------------------------------------------------------------------
# C2 的静态搜索验法（可选，U-5）
# ---------------------------------------------------------------------------
# 标识符按原样（区分大小写）匹配，带前后缀的也算（m_ov_months、row_L_label）。裸的 L 和阶梯层数 L 撞名，默认不搜，
# 要搜就用 extra_identifiers=("L",)。L_label 的三个取值另搜：条件里拿它们比较，不管变量叫什么都算。
FORBIDDEN_IDENTIFIERS = ("L_label", "m_ov")
L_LABEL_VALUES = L_LABELS


@dataclass(frozen=True)
class BranchHit:
    """一处以禁用量为条件的分支。"""
    path: str
    line: int
    kind: str                 # if / elif / 条件表达式 / while / assert / 推导式 if / match / 无法解析
    names: tuple[str, ...]    # 命中的标识符或 L_label 取值
    code: str                 # 那一行源码


def _matches(name: str, ident: str) -> bool:
    return re.search(rf"(?:^|_){re.escape(ident)}(?:_|$)", name) is not None


def _names_in(expr: ast.AST) -> Iterable[str]:
    """条件表达式里的变量名、属性名、关键字参数名和字符串常量（row['m_ov'] 这种也算）。"""
    for sub in ast.walk(expr):
        if isinstance(sub, ast.Name):
            yield sub.id
        elif isinstance(sub, ast.Attribute):
            yield sub.attr
        elif isinstance(sub, ast.keyword) and sub.arg:
            yield sub.arg
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub.value


def _conditions(tree: ast.AST) -> Iterable[tuple[ast.AST, str]]:
    """所有分支条件：if / elif、三元式、while、assert、推导式里的 if、match 的主语。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            yield node.test, "if"
        elif isinstance(node, ast.IfExp):
            yield node.test, "条件表达式"
        elif isinstance(node, ast.While):
            yield node.test, "while"
        elif isinstance(node, ast.Assert):
            yield node.test, "assert"
        elif isinstance(node, ast.comprehension):
            for cond in node.ifs:
                yield cond, "推导式 if"
        elif isinstance(node, ast.Match):
            yield node.subject, "match"


def _iter_py(paths) -> Iterable[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*.py") if "__pycache__" not in q.parts)
        elif p.suffix == ".py":
            yield p


def static_branch_check(paths, identifiers: Iterable[str] = FORBIDDEN_IDENTIFIERS, *,
                        extra_identifiers: Iterable[str] = (), check_label_values: bool = True) -> list[BranchHit]:
    """扫源码，报出以 L_label、m_ov（或 L_label 的取值）为条件的分支（C2 的静态搜索验法，可选）。返回空列表 = 通过。

    paths 可给文件或目录（目录递归找 *.py）。只看语法树里的分支条件，注释、文档串、普通赋值、打印、查表都不算。
    """
    idents = tuple(identifiers) + tuple(extra_identifiers)
    hits: list[BranchHit] = []
    for path in _iter_py(paths):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as err:
            hits.append(BranchHit(str(path), err.lineno or 0, "无法解析", (), str(err)))
            continue
        for cond, kind in _conditions(tree):
            found = []
            for name in _names_in(cond):
                found += [i for i in idents if _matches(name, i)]
                if check_label_values and name in L_LABEL_VALUES:
                    found.append(repr(name))
            if not found:
                continue
            code = lines[cond.lineno - 1].strip() if 0 < cond.lineno <= len(lines) else ""
            if kind == "if" and code.startswith("elif"):
                kind = "elif"
            hits.append(BranchHit(str(path), cond.lineno, kind, tuple(dict.fromkeys(found)), code))
    return sorted(hits, key=lambda h: (h.path, h.line))


def format_hits(hits: list[BranchHit]) -> str:
    """把检查结果写成人读的几行。"""
    if not hits:
        return "静态分支检查通过：没有以 L_label / m_ov 为条件的分支（C2）。"
    rows = [f"{h.path}:{h.line} [{h.kind}] 条件用到 {', '.join(h.names)} → {h.code}" for h in hits]
    return f"静态分支检查不过（{len(hits)} 处，C2）：\n" + "\n".join(rows)


if __name__ == "__main__":  # python -m stockburger.records <文件或目录>…（默认扫本包）
    targets = sys.argv[1:] or [str(Path(__file__).resolve().parent)]
    found = static_branch_check(targets)
    print(format_hits(found))
    sys.exit(1 if found else 0)
