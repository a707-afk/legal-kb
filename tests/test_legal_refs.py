"""法条引用解析测试（D-07 的地基）。

核心契约：**中文数字与阿拉伯数字必须归一化后比较**。
不做归一化会双向出错：判分侧把正确引用判成幻觉（实测条号精度 1.0 → 0.781），
校验侧（D-07）大量误杀合法引用。
"""
from __future__ import annotations

from src.legal_refs import (
    article_hit,
    cn2arabic,
    extract_citations,
    normalize_article_text,
)


class TestCn2Arabic:
    def test_units(self):
        assert cn2arabic("一") == 1
        assert cn2arabic("九") == 9
        assert cn2arabic("十") == 10
        assert cn2arabic("百") == 100
        assert cn2arabic("千") == 1000

    def test_compounds(self):
        assert cn2arabic("十一") == 11
        assert cn2arabic("二十") == 20
        assert cn2arabic("二十三") == 23
        assert cn2arabic("三十八") == 38
        assert cn2arabic("一百零五") == 105
        assert cn2arabic("五百七十七") == 577
        assert cn2arabic("一千二百三十四") == 1234
        assert cn2arabic("一千零八十八") == 1088

    def test_arabic_and_fullwidth(self):
        assert cn2arabic("577") == 577
        assert cn2arabic("０１２") == 12

    def test_invalid_returns_none(self):
        assert cn2arabic("") is None
        assert cn2arabic("abc") is None
        assert cn2arabic("第一条") is None


class TestNormalize:
    def test_cn_article_to_arabic(self):
        assert normalize_article_text("第五百七十七条") == "第577条"
        assert normalize_article_text("第三十八条") == "第38条"
        assert normalize_article_text("第十条") == "第10条"

    def test_already_arabic_unchanged(self):
        assert normalize_article_text("第577条") == "第577条"

    def test_mixed_text(self):
        out = normalize_article_text("依《民法典》第五百七十七条与第584条")
        assert "第577条" in out and "第584条" in out

    def test_item_and_paren(self):
        # 括号形式保留括号（`（二）` → `（2）`），不强行改成"第2项"
        assert normalize_article_text("第三十八条第（二）项") == "第38条第（2）项"
        assert normalize_article_text("第38条第2款") == "第38条第2款"
        assert normalize_article_text("第三十八条第（二）项") .count("第38条") == 1


class TestExtractCitations:
    def test_law_binding(self):
        cits = extract_citations("根据《中华人民共和国劳动合同法》第三十八条规定，可以解除。")
        assert len(cits) == 1
        assert cits[0].law == "中华人民共和国劳动合同法"
        assert cits[0].article == 38

    def test_item_bound_after_article(self):
        cits = extract_citations("《劳动合同法》第三十八条第（二）项规定未及时足额支付劳动报酬。")
        assert cits[0].article == 38
        assert cits[0].item == 2
        assert cits[0].item_kind == "项"

    def test_multiple_articles_one_law(self):
        cits = extract_citations("《劳动合同法》第三十八条……第四十六条规定应当支付经济补偿。")
        arts = [c.article for c in cits]
        assert arts == [38, 46]
        assert all(c.law == "劳动合同法" for c in cits)

    def test_default_law_fallback(self):
        cits = extract_citations("第三十八条可以解除。", default_law="劳动合同法")
        assert cits[0].law == "劳动合同法"

    def test_no_citation(self):
        assert extract_citations("这是一段没有条号的文字。") == []


class TestArticleHit:
    def test_arabic_answer_matches_chinese_context(self):
        """答案写 `第577条`，语料写 `第五百七十七条` —— 必须判命中。"""
        cits = extract_citations("《民法典》第577条规定应当承担违约责任。")
        hit, total = article_hit(cits, "《中华人民共和国民法典》第五百七十七条：当事人一方不履行合同义务……")
        assert (hit, total) == (1, 1)

    def test_chinese_answer_matches_chinese_context(self):
        cits = extract_citations("《民法典》第五百七十七条规定。")
        hit, total = article_hit(cits, "第五百七十七条：当事人一方不履行……")
        assert (hit, total) == (1, 1)

    def test_hallucinated_article_not_hit(self):
        """答案引用了片段里没有的条号 —— 必须判未命中（这是引用校验的核心）。"""
        cits = extract_citations("《民法典》第999条规定。")
        hit, total = article_hit(cits, "第五百七十七条：当事人一方不履行……")
        assert (hit, total) == (0, 1)

    def test_partial_hit(self):
        cits = extract_citations("《民法典》第577条与第999条规定。")
        hit, total = article_hit(cits, "第五百七十七条：……")
        assert (hit, total) == (1, 2)
