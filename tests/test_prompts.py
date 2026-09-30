"""文本级回归测试：确保关键 prompt 规则不被误删或漂移。

这些测试不调用 LLM，只是纯文本断言（assert "关键词" in PROMPT），
运行快、零成本。任何改动导致关键规则丢失都会在此拦截。
"""
from src.agent.prompts import (
    EVALUATOR_SYSTEM_PROMPT,
    GENERATOR_RULES,
    GENERATOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIZE_RULES,
    SYNTHESIZE_SYSTEM_PROMPT,
)


# ─────────────────────────────────────────────────────────────────────────────
# 生成器 prompt 回归测试
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneratorPrompt:
    """生成器 system prompt 必须包含的关键规则。"""

    def test_currency_branch_not_yet_effective(self):
        """时效三分支：尚未生效。"""
        assert "尚未生效" in GENERATOR_SYSTEM_PROMPT

    def test_currency_branch_repealed(self):
        """时效三分支：已废止。"""
        assert "已废止" in GENERATOR_SYSTEM_PROMPT

    def test_currency_branch_unlabeled(self):
        """时效三分支：未标注。"""
        assert "未标注" in GENERATOR_SYSTEM_PROMPT

    def test_citation_numbering_rule(self):
        """引用编号规则 [n]。"""
        assert "[1][2]" in GENERATOR_SYSTEM_PROMPT
        assert "证据编号" in GENERATOR_SYSTEM_PROMPT

    def test_no_legal_opinion_disclaimer(self):
        """不构成法律意见声明。"""
        assert "不构成法律意见" in GENERATOR_SYSTEM_PROMPT

    def test_no_internal_system_exposure(self):
        """不暴露内部系统规则。"""
        assert "不要暴露内部系统" in GENERATOR_SYSTEM_PROMPT

    def test_insufficient_evidence_declaration(self):
        """证据不足时必须声明。"""
        assert "未检索到相关内容" in GENERATOR_SYSTEM_PROMPT
        assert "超出本知识库范围" in GENERATOR_SYSTEM_PROMPT

    def test_no_fabrication_rule(self):
        """禁止编造法条。"""
        assert "不要编造法条" in GENERATOR_SYSTEM_PROMPT

    def test_excerpt_original_text(self):
        """引用法条时直接摘录原文。"""
        assert "直接摘录证据中的原文" in GENERATOR_SYSTEM_PROMPT

    def test_no_decoration_rule(self):
        """不要输出无关章节装饰。"""
        assert "不要输出与证据无关的章节标题" in GENERATOR_SYSTEM_PROMPT

    def test_generator_has_8_rules(self):
        """生成器规则列表恰好 8 条。"""
        assert len(GENERATOR_RULES) == 8


# ─────────────────────────────────────────────────────────────────────────────
# 评估器 prompt 回归测试
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluatorPrompt:
    """评估器 system prompt 必须包含的关键规则。"""

    def test_grounded_judgment_rule(self):
        """grounding 判定规则。"""
        assert "grounded=true" in EVALUATOR_SYSTEM_PROMPT
        assert "grounded=false" in EVALUATOR_SYSTEM_PROMPT

    def test_honest_declaration_not_unsupported(self):
        """诚实声明未检索到不算 unsupported。"""
        assert "未检索到" in EVALUATOR_SYSTEM_PROMPT
        assert "诚实表述" in EVALUATOR_SYSTEM_PROMPT
        assert "不算 unsupported" in EVALUATOR_SYSTEM_PROMPT

    def test_fabrication_triggers_false(self):
        """编造条号/日期/内容才判 false。"""
        assert "编造条号" in EVALUATOR_SYSTEM_PROMPT
        assert "生效日期" in EVALUATOR_SYSTEM_PROMPT
        assert "条文内容" in EVALUATOR_SYSTEM_PROMPT

    def test_json_output_format(self):
        """要求返回 JSON。"""
        assert "unsupported_claims" in EVALUATOR_SYSTEM_PROMPT
        assert "safe" in EVALUATOR_SYSTEM_PROMPT
        assert "只返回 JSON" in EVALUATOR_SYSTEM_PROMPT

    def test_safety_check(self):
        """安全检查：不暴露内部系统名、API key。"""
        assert "内部系统名" in EVALUATOR_SYSTEM_PROMPT
        assert "API key" in EVALUATOR_SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# 规划器 prompt 回归测试
# ─────────────────────────────────────────────────────────────────────────────

class TestPlannerPrompt:
    """规划器 system prompt 必须包含的关键规则。"""

    def test_json_array_output_format(self):
        """JSON 数组输出格式要求（步骤数组）。"""
        assert "步骤数组" in PLANNER_SYSTEM_PROMPT
        assert "type" in PLANNER_SYSTEM_PROMPT
        assert "tool" in PLANNER_SYSTEM_PROMPT
        assert "params" in PLANNER_SYSTEM_PROMPT

    def test_retrieve_type_explained(self):
        """retrieve 类型说明。"""
        assert "retrieve" in PLANNER_SYSTEM_PROMPT
        assert "检索本地法律法规知识库" in PLANNER_SYSTEM_PROMPT

    def test_execute_type_explained(self):
        """execute 类型说明。"""
        assert "execute" in PLANNER_SYSTEM_PROMPT
        assert "调用工具" in PLANNER_SYSTEM_PROMPT

    def test_no_invented_tools(self):
        """不要发明不存在的工具。"""
        assert "不要发明不存在的工具" in PLANNER_SYSTEM_PROMPT

    def test_no_placeholder_params(self):
        """params 不要用占位符。"""
        assert "不要用占位符" in PLANNER_SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# synthesize prompt 回归测试
# ─────────────────────────────────────────────────────────────────────────────

class TestSynthesizePrompt:
    """synthesize 工具 system prompt 必须包含的关键规则。"""

    def test_shares_citation_rule_with_generator(self):
        """与生成器共享引用标注规则。"""
        assert "[1][2]" in SYNTHESIZE_SYSTEM_PROMPT
        assert "证据编号" in SYNTHESIZE_SYSTEM_PROMPT

    def test_shares_no_fabrication_rule(self):
        """与生成器共享禁编造规则。"""
        assert "不要编造法条" in SYNTHESIZE_SYSTEM_PROMPT

    def test_shares_insufficient_evidence_rule(self):
        """与生成器共享证据不足声明规则。"""
        assert "未检索到相关内容" in SYNTHESIZE_SYSTEM_PROMPT

    def test_shares_currency_rule(self):
        """与生成器共享时效三分支规则。"""
        assert "尚未生效" in SYNTHESIZE_SYSTEM_PROMPT
        assert "已废止" in SYNTHESIZE_SYSTEM_PROMPT
        assert "未标注" in SYNTHESIZE_SYSTEM_PROMPT

    def test_shares_disclaimer_rule(self):
        """与生成器共享不构成法律意见声明。"""
        assert "不构成法律意见" in SYNTHESIZE_SYSTEM_PROMPT

    def test_synthesize_has_5_rules(self):
        """synthesize 规则列表恰好 5 条。"""
        assert len(SYNTHESIZE_RULES) == 5


# ─────────────────────────────────────────────────────────────────────────────
# 规则一致性测试：确保 synthesize 规则是生成器规则的子集
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleConsistency:
    """验证 synthesize 规则确实引用了生成器的权威源（无漂移）。"""

    def test_synthesize_rules_are_subset_of_generator(self):
        """synthesize 每条规则都能在生成器规则列表中找到完全匹配的文本。"""
        for rule in SYNTHESIZE_RULES:
            assert rule in GENERATOR_RULES, (
                f"synthesize 规则漂移！以下规则不在生成器权威源中：\n{rule[:80]}..."
            )

    def test_shared_rules_identical_text(self):
        """共享规则的文本完全相同（不是相似，是相同对象引用）。"""
        shared_indices = []
        for syn_rule in SYNTHESIZE_RULES:
            for i, gen_rule in enumerate(GENERATOR_RULES):
                if syn_rule is gen_rule:  # 同一对象引用
                    shared_indices.append(i)
                    break
        # synthesize 有 5 条规则，应能在生成器中找到 5 个对应
        assert len(shared_indices) == 5
