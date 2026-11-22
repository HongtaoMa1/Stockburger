# -*- coding: utf-8 -*-
# Required Notice: Copyright (c) 2026 Hongtao Ma (Rick) (https://github.com/HongtaoMa1/Stockburger)
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE-CODE) with the additional permission in README.md.
"""审计账的单测（v007 C1 / C2）。"""
import json
from datetime import date
from pathlib import Path

import pytest

from stockburger import records as R
from stockburger import toplayer_prompt as T

PKG_DIR = Path(R.__file__).resolve().parent


def _r0_spec(ban=True):
    """原型唯一的提示（R0）实际发出时的配置。"""
    fams = [T.FamilySchema("MOM", "rank of past return"), T.FamilySchema("VAL", "rank of book-to-price")]
    return T.build_schema_only_prompt("跑赢沪深 300", fams, ("2025-01-02", "2025-12-31"), ban=ban)[1]


# ---- 截止月、L_label、m_ov --------------------------------------------------
@pytest.mark.parametrize("reported, expected", [
    ("June 2025", date(2025, 6, 30)),
    ("Jun 2025", date(2025, 6, 30)),
    ("2025-06", date(2025, 6, 30)),
    ("2025年2月", date(2025, 2, 28)),
    ("Feb 2024", date(2024, 2, 29)),
    (date(2025, 6, 15), date(2025, 6, 30)),
])
def test_t_c_takes_month_end(reported, expected):
    assert R.month_end(reported) == expected


def test_month_end_rejects_garbage():
    with pytest.raises(ValueError):
        R.month_end("sometime soon")


def test_label_three_cases_by_date():
    t_c = date(2025, 6, 30)
    assert R.cutoff_label(t_c, date(2025, 7, 1), date(2025, 11, 13)) == "clean"         # t_c < t_1
    assert R.cutoff_label(t_c, date(2025, 3, 1), date(2025, 11, 13)) == "partial"       # t_1 ≤ t_c < t_2
    assert R.cutoff_label(t_c, date(2025, 6, 30), date(2025, 11, 13)) == "partial"      # t_1 = t_c 仍是 partial
    assert R.cutoff_label(t_c, date(2025, 1, 1), date(2025, 6, 30)) == "contaminated"   # t_c ≥ t_2
    with pytest.raises(ValueError):
        R.cutoff_label(t_c, date(2025, 9, 1), date(2025, 8, 1))


def test_m_ov_formula():
    t_c = date(2025, 6, 30)
    assert R.overlap_months(t_c, date(2025, 3, 1), date(2025, 11, 13)) == pytest.approx(121 / 30.4375)   # partial
    assert R.overlap_months(t_c, date(2025, 7, 1), date(2025, 11, 13)) == 0.0                            # clean
    assert R.overlap_months(t_c, date(2025, 1, 1), date(2025, 5, 31)) == pytest.approx(150 / 30.4375)    # 整段


# ---- 审计账一行（C1）-----------------------------------------------------------
def test_record_has_exactly_the_c1_fields():
    rec = R.make_call_record("example-llm (build 2026-01)", "2026-01-15", "June 2025",
                             "2025-01-02", "2025-12-31", prompt=_r0_spec())
    assert rec.model_id == "example-llm (build 2026-01)"                          # 原串不改写
    assert (rec.call_date, rec.t_c) == (date(2026, 1, 15), date(2025, 6, 30))
    assert (rec.t_1, rec.t_2) == (date(2025, 1, 2), date(2025, 12, 31))
    assert rec.L_label == "partial" and rec.m_ov == pytest.approx(R.overlap_months(rec.t_c, rec.t_1, rec.t_2))
    assert rec.ban_variant == "W2" and rec.ban_hash == T.sentence_hash(T.W2_SENTENCE)
    row = json.loads(json.dumps(rec.to_json(), ensure_ascii=False))
    assert set(row) == {"model_id", "call_date", "t_c", "t_1", "t_2", "m_ov", "L_label", "ban_variant", "ban_hash"}
    assert row["L_label"] == "partial" and row["t_c"] == "2025-06-30" and row["ban_variant"] == "W2"
    text = rec.render()
    assert "L_label = partial" in text and "m_ov = " in text and "禁令句 = 开 · W2；句子哈希 " in text


def test_ban_off_and_unrecorded():
    off = R.make_call_record("m", "2026-01-15", "2025-06", "2025-01-02", "2025-12-31", prompt=_r0_spec(ban=False))
    assert (off.ban_variant, off.ban_hash) == ("off", None) and "禁令句 = 关" in off.render()
    blank = R.make_call_record("m", "2026-01-15", "2025-06", "2025-01-02", "2025-12-31")
    assert (blank.ban_variant, blank.ban_hash) == (None, None) and "禁令句 = 未记" in blank.render()
    r1 = R.make_call_record("m", "2026-01-15", "2025-06", "2025-01-02", "2025-12-31", prompt=T.PromptSpec("R1+", "W1"))
    assert r1.ban_hash == T.sentence_hash(T.W1_SENTENCE)
    with pytest.raises(TypeError):
        R.make_call_record("m", "2026-01-15", "2025-06", "2025-01-02", "2025-12-31", prompt={"ban_variant": "W2"})


def test_test_score_tag_by_lookup():
    """C1：L_label ≠ clean 时日报里测试段成绩旁标「压力测试 / 概念验证级」。"""
    clean = R.make_call_record("m", "2026-01-15", "2024-06", "2025-01-02", "2025-12-31")
    part = R.make_call_record("m", "2026-01-15", "2025-06", "2025-01-02", "2025-12-31")
    dirty = R.make_call_record("m", "2026-01-15", "2026-03", "2025-01-02", "2025-12-31")
    assert [r.L_label for r in (clean, part, dirty)] == ["clean", "partial", "contaminated"]
    assert (clean.test_tag, part.test_tag, dirty.test_tag) == ("", R.STRESS_TAG, R.STRESS_TAG)
    assert R.TEST_SCORE_TAG[None] == ""                                                 # 顶层没接上：没有审计行，不标


def test_append_jsonl_only_appends(tmp_path):
    path = tmp_path / "audit" / "ledger.jsonl"
    r1 = R.make_call_record("a", "2026-01-15", "2025-06", "2025-01-02", "2025-12-31").to_json()
    r2 = R.make_call_record("b", "2026-01-16", "2024-06", "2025-01-02", "2025-12-31").to_json()
    R.append_jsonl(path, r1)
    first = path.read_text(encoding="utf-8")
    R.append_jsonl(path, r2)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and lines[0] + "\n" == first
    assert json.loads(lines[1])["L_label"] == "clean"


# ---- C2 的静态搜索验法（可选）-------------------------------------------------
BAD_SRC = '''\
def gate(row, m_ov, L_label):
    if m_ov > 3:
        return "stop"
    elif L_label == "partial":
        return "warn"
    x = 1 if row["m_ov"] else 0
    while L_label:
        break
    assert m_ov_months < 2
    ok = [r for r in row if r.L_label]
    if row.status == "contaminated":
        pass
    return x, ok
'''

CLEAN_SRC = '''\
"""m_ov 写在文档串里不算。"""
TAG = {"clean": "", "partial": "压力测试 / 概念验证级"}


def f(m_ov, L_label, probe):
    # if m_ov > 3:  注释不算
    total = m_ov + 1
    print(L_label, total, TAG[L_label])          # 查表印标签不算分支（C1）
    if probe is None:                            # 缺值检查，不是禁用量
        return {"m_ov": m_ov}
    return {"L_label": total}
'''


def test_static_check_flags_each_forbidden_branch(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text(BAD_SRC, encoding="utf-8")
    hits = R.static_branch_check([p])
    assert {(h.line, h.kind) for h in hits} == {(2, "if"), (4, "elif"), (6, "条件表达式"), (7, "while"), (9, "assert"),
                                               (10, "推导式 if"), (11, "if")}
    by_line = {h.line: h.names for h in hits}
    assert by_line[2] == ("m_ov",) and set(by_line[4]) == {"L_label", "'partial'"}
    assert by_line[9] == ("m_ov",)                                       # m_ov_months 这种带后缀的也算
    assert by_line[11] == ("'contaminated'",)                            # 拿 L_label 的取值比较就算
    assert "不过" in R.format_hits(hits)


def test_static_check_ignores_non_branch_uses(tmp_path):
    (tmp_path / "clean.py").write_text(CLEAN_SRC, encoding="utf-8")
    assert R.static_branch_check(tmp_path) == []
    assert "通过" in R.format_hits([])


def test_static_check_allows_selecting_ban_by_prompt_type(tmp_path):
    """C1 / C2：按提示里有没有表选禁令句不算分支。"""
    p = tmp_path / "select.py"
    p.write_text('def pick(prompt_type, ban):\n    if prompt_type == "R0":\n        return "W2" if ban else "off"\n'
                 '    return "W1"\n', encoding="utf-8")
    assert R.static_branch_check([p]) == []


def test_static_check_bare_L_only_on_request(tmp_path):
    p = tmp_path / "ladder.py"
    p.write_text("def f(L):\n    if L == 0:\n        return 1\n", encoding="utf-8")
    assert R.static_branch_check([p]) == []                               # 裸 L 与阶梯层数撞名，默认不搜
    assert len(R.static_branch_check([p], extra_identifiers=("L",))) == 1


def test_static_check_reports_unparsable_file(tmp_path):
    p = tmp_path / "broken.py"
    p.write_text("if m_ov >\n", encoding="utf-8")
    hits = R.static_branch_check([p])
    assert len(hits) == 1 and hits[0].kind == "无法解析"


def test_my_modules_have_no_forbidden_branch():
    mine = [PKG_DIR / f"{name}.py" for name in ("intake", "records", "toplayer_prompt", "product")]
    assert R.static_branch_check(mine) == [], R.format_hits(R.static_branch_check(mine))


def test_whole_package_has_no_forbidden_branch():
    # 全包只按标识符搜（L_label 取值的启发式只用在上面几个模块，免得误伤别的模块）
    hits = R.static_branch_check(PKG_DIR, check_label_values=False)
    assert hits == [], R.format_hits(hits)
