"""Qwen3-Reranker（CausalLM yes/no logits），与 sentence-transformers CrossEncoder 不兼容。"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from llama_index.core.schema import NodeWithScore
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# CPU 上若按 8192 全序列 padding，CausalLM 的 lm_head 会产生数 GB logits 导致 OOM；需在 CPU 收紧上限。
_CPU_RERANK_MAX_LENGTH = int(os.environ.get("QWEN_RERANK_CPU_MAX_LENGTH", "1024"))

_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# 与官方 README 一致的特殊符号（tokenizer 中 im_end 记为 <|im_end|>）
_IM_END = "<|im_end|>"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class _QwenRerankRuntime:
    __slots__ = (
        "tokenizer",
        "model",
        "token_true_id",
        "token_false_id",
        "prefix_tokens",
        "suffix_tokens",
        "max_length",
        "device",
    )

    def __init__(self, model_path: str, max_length: int, device: str) -> None:
        path = str(Path(model_path).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, padding_side="left", trust_remote_code=True
        )
        dev = torch.device(device)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, dtype=dtype
        ).eval()
        self.device = dev
        self.model.to(self.device)
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        if self.device.type == "cpu":
            eff = min(max_length, _CPU_RERANK_MAX_LENGTH)
            if eff < max_length:
                logger.warning(
                    "Qwen3-Reranker 在 CPU 上将 max_length %s 收紧为 %s，避免 lm_head OOM",
                    max_length,
                    eff,
                )
            self.max_length = eff
        else:
            self.max_length = max_length

        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query "
            "and the Instruct provided. Note that the answer can only be "
            '"yes" or "no".'
            f"{_IM_END}\n<|im_start|>user\n"
        )
        suffix = (
            f"{_IM_END}\n<|im_start|>assistant\n"
            f"{_THINK_OPEN}\n\n{_THINK_CLOSE}\n\n"
        )
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        logger.info(
            "Qwen3-Reranker 已加载: %s device=%s max_length=%s",
            path,
            self.device,
            self.max_length,
        )

    def _format_pair(self, instruction: str | None, query: str, doc: str) -> str:
        inst = instruction or _DEFAULT_INSTRUCTION
        return (
            f"<Instruct>: {inst}\n<Query>: {query}\n<Document>: {doc}"
        )

    def _process_batch(self, pairs: list[str]) -> dict:
        tok = self.tokenizer
        max_len = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        inputs = tok(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_len,
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        # 按 batch 内最长序列 padding，避免短样本也被 pad 到 max_length 白白占满 logits 形状
        inputs = tok.pad(
            inputs,
            padding="longest",
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.inference_mode()
    def score_pairs(self, pairs: list[str], batch_size: int) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            inputs = self._process_batch(batch)
            logits = self.model(**inputs).logits[:, -1, :]
            true_v = logits[:, self.token_true_id]
            false_v = logits[:, self.token_false_id]
            stacked = torch.stack([false_v, true_v], dim=1)
            probs = torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp()
            scores.extend([float(x) for x in probs.tolist()])
        return scores


@lru_cache(maxsize=4)
def _runtime(model_path: str, max_length: int, device: str) -> _QwenRerankRuntime:
    return _QwenRerankRuntime(model_path, max_length, device)


def rerank_nodes_qwen(
    query: str,
    scored: list[NodeWithScore],
    top_n: int,
    model_path: str,
    *,
    max_length: int = 8192,
    batch_size: int = 4,
    instruction: str | None = None,
    inference_device: str | None = None,
) -> list[NodeWithScore]:
    if not scored or top_n < 1:
        return []
    from src.config_LEGACY_REFERENCE import get_settings
    from src.inference_device import resolve_inference_device

    settings = get_settings()
    dev = inference_device or resolve_inference_device(settings)
    rt = _runtime(str(Path(model_path).resolve()), max_length, dev)
    pairs = [
        rt._format_pair(instruction, query, sn.node.get_content() or "")
        for sn in scored
    ]
    # CPU 上 batch>1 时 lm_head 仍易峰值过高，默认串行
    bs = batch_size if rt.device.type == "cuda" else 1
    raw_scores = rt.score_pairs(pairs, batch_size=bs)
    combined = list(zip(scored, raw_scores))
    combined.sort(key=lambda x: float(x[1]), reverse=True)
    out: list[NodeWithScore] = []
    for sn, s in combined[:top_n]:
        out.append(NodeWithScore(node=sn.node, score=float(s)))
    return out


def smoke_test_score(model_path: str) -> float:
    """返回单条 (query, doc) 的相关分，用于启动自检。"""
    from llama_index.core.schema import TextNode

    from src.config_LEGACY_REFERENCE import get_settings
    from src.inference_device import resolve_inference_device

    node = TextNode(text="中国的首都是北京。", id_="smoke")
    ns = NodeWithScore(node=node, score=1.0)
    out = rerank_nodes_qwen(
        "中国的首都是哪里？",
        [ns],
        top_n=1,
        model_path=model_path,
        batch_size=1,
        inference_device=resolve_inference_device(get_settings()),
    )
    return float(out[0].score) if out else -1.0
