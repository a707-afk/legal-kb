"""front-matter 解析的类型契约测试（D1 配套）。

背景：`status_code` 是时效治理（D-02 / D-06）的判据字段。front-matter 是纯文本，
若不显式转换，`status_code: 3` 会变成字符串 "3"、`status_code: null` 会变成 "null"，
下游 `int()` / `== 3` 会**静默出错**——这正是旧项目"字段名 bug 让整个时效治理失效"
（HANDOFF §9.1）的同类事故。
"""
from __future__ import annotations

from src.chunking import _parse_frontmatter


def _fm(**kwargs) -> str:
    lines = ["---"]
    lines += [f"{k}: {v}" for k, v in kwargs.items()]
    lines += ["---", "", "# 标题", "", "正文。"]
    return "\n".join(lines)


def test_status_code_int_is_int():
    meta, _ = _parse_frontmatter(_fm(status_code=3, title='"某法"'))
    assert meta["status_code"] == 3
    assert isinstance(meta["status_code"], int)


def test_status_code_null_is_none():
    meta, _ = _parse_frontmatter(_fm(status_code="null", title='"某法"'))
    assert meta["status_code"] is None


def test_status_code_empty_is_none():
    meta, _ = _parse_frontmatter(_fm(status_code="", title='"某法"'))
    assert meta["status_code"] is None


def test_status_code_negative_one_preserved():
    """-1 = 未知，必须保留为 -1，不能当成"缺失"。"""
    meta, _ = _parse_frontmatter(_fm(status_code=-1, title='"某法"'))
    assert meta["status_code"] == -1


def test_version_not_coerced_to_int():
    """version 是 bbbs（可能全为数字），**不能**被转成 int（会丢前导零）。"""
    meta, _ = _parse_frontmatter(_fm(version='"0012345678"', status_code=3))
    assert meta["version"] == "0012345678"
    assert isinstance(meta["version"], str)


def test_quoted_and_unquoted_values():
    meta, _ = _parse_frontmatter(_fm(
        title='"中华人民共和国人民法院组织法"',
        effective_date='"2020-01-01"',
        source_category="法律",
        status_code=3,
    ))
    assert meta["title"] == "中华人民共和国人民法院组织法"
    assert meta["effective_date"] == "2020-01-01"
    assert meta["source_category"] == "法律"


def test_missing_status_code_key_absent():
    """没有该键时不凭空造一个，交由调用方判断。"""
    meta, _ = _parse_frontmatter(_fm(title='"某法"'))
    assert "status_code" not in meta


def test_body_strips_frontmatter():
    _, body = _parse_frontmatter(_fm(status_code=3, title='"某法"'))
    assert body.startswith("# 标题")
    assert "status_code" not in body
