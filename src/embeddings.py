"""统一 Embedding 封装。

优先用 sentence-transformers 加载本地 bge 系列模型，回退到
llama-index HuggingFaceEmbedding；两者均支持本地快照、零网络加载，
并对外提供 sync / async 的 encode 接口。
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any, List, Optional

from src.config_LEGACY_REFERENCE import get_settings
from src.inference_device import log_device_context, resolve_inference_device

logger = logging.getLogger(__name__)

_KNOWN_ST_MODELS = frozenset({
    "BAAI/bge-m3",
    "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-small-zh-v1.5",
    "BAAI/bge-base-zh-v1.5",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-base",
    "intfloat/e5-mistral-7b-instruct",
    "maidalun1020/bce-embedding-base_v1",
})


def _is_sentence_transformers_model(name: str) -> bool:
    """判断模型名是否应走 sentence-transformers 加载路径。"""
    lower = name.lower()
    if any(known.lower() in lower for known in _KNOWN_ST_MODELS):
        return True
    if "/bge-" in lower or "bge-" in lower:
        return True
    if "bce-embedding" in lower:
        return True
    if "multilingual-e5" in lower:
        return True
    return False


@lru_cache(maxsize=2)
def _load_st_model(model_path: str, device: str, fp16: bool = False):
    """加载 sentence-transformers 模型（传入本地目录时不产生任何网络请求）。

    fp16：只在 device=cuda 时生效。bge-m3 权重 fp32 = 2.27GB，8GB 显存机器上
    和 reranker 挤不下；fp16 减半权重且吞吐约 2 倍（实测 fp32 + batch=32 时
    GPU 只吃到 74W/220W，根本喂不饱）。
    """
    from sentence_transformers import SentenceTransformer

    log_device_context(f"ST: {model_path}", device)
    model = SentenceTransformer(model_path, device=device)
    if fp16 and device == "cuda":
        try:
            model.half()
            logger.info("ST 模型已转 fp16: %s", model_path)
        except Exception:  # noqa: BLE001
            logger.warning("fp16 转换失败，沿用 fp32（不影响正确性，只影响速度）")
    elif fp16:
        logger.info("device=%s 非 cuda，忽略 fp16 请求", device)
    return model


@lru_cache(maxsize=2)
def _load_hf_model(model_name: str, device: str):
    """加载 llama-index HuggingFaceEmbedding（本地目录同样零网络）。"""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    log_device_context(f"HF: {model_name}", device)
    return HuggingFaceEmbedding(
        model_name=model_name,
        trust_remote_code=True,
        device=device,
    )


def _hf_hub_cache_bases():
    """候选 HF hub 缓存根目录（HF_HUB_CACHE > HF_HOME/hub > ~/.cache/huggingface/hub）。"""
    import os
    from pathlib import Path

    bases = []
    if os.environ.get("HF_HUB_CACHE"):
        bases.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        bases.append(Path(os.environ["HF_HOME"]) / "hub")
    bases.append(Path.home() / ".cache" / "huggingface" / "hub")
    return bases


def resolve_local_hf_snapshot(model_name: str) -> str | None:
    """模型已在本地 HF 缓存时返回快照目录；否则返回 None。

    背景（docs/FIX-retrieval-hang-20260927.md）：即使模型完整缓存，
    SentenceTransformer 按仓库名加载时仍会对 huggingface.co 发 HEAD 版本检查；
    无外网时每次 TCP connect ~21s（WinError 10060）× 每文件重试 5 次指数退避
    × 多个探测文件，首次检索被阻塞数分钟。返回本地快照目录后走纯本地加载，
    零网络、亚秒级。
    """

    if "/" not in model_name:
        return None
    repo_dirname = "models--" + model_name.replace("/", "--")
    for base in _hf_hub_cache_bases():
        hub_dir = base / repo_dirname
        if not hub_dir.is_dir():
            continue
        snap_root = hub_dir / "snapshots"
        if not snap_root.is_dir():
            continue
        # 优先 refs/main 指向的 revision
        ref = hub_dir / "refs" / "main"
        if ref.is_file():
            rev = ref.read_text(encoding="utf-8").strip()
            snap = snap_root / rev
            if snap.is_dir() and any(snap.iterdir()):
                return str(snap)
        # 兜底：恰好一个非空快照
        snaps = [d for d in snap_root.iterdir() if d.is_dir() and any(d.iterdir())]
        if len(snaps) == 1:
            return str(snaps[0])
    return None


class EmbeddingModel:
    """统一的 Embedding 模型封装。

    - 自动在 sentence-transformers 与 HuggingFace 后端间选择
    - 提供 async / sync 两套 encode 接口
    - 本地 HF 快照加载，查询路径零网络（防挂死）
    - 模型对象经 lru_cache 复用
    """

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        device: str = "cpu",
        expected_dimension: Optional[int] = None,
        fp16: bool = False,
    ) -> None:
        self._model_name = model_name
        self._model_path = model_path or model_name
        self._device = device
        self._fp16 = bool(fp16)
        self._resolved_path: Optional[str] = None
        # 维度断言（D2）：加载后校验，不匹配直接抛错，**不静默回退**
        self._expected_dimension = expected_dimension
        self._checked = False

        import os
        if _is_sentence_transformers_model(model_name):
            local_path = model_path or ""
            if local_path and os.path.isdir(local_path):
                self._resolved_path = local_path
            elif os.path.isdir(model_name):
                self._resolved_path = model_name
            else:
                # 已缓存到本地 HF hub → 解析快照目录，查询路径零网络（防挂死）
                snapshot = resolve_local_hf_snapshot(model_name)
                self._resolved_path = snapshot or model_name
                if snapshot:
                    logger.info(
                        "Embedding 模型使用本地 HF 快照: %s (无网络请求)", snapshot
                    )
            self._st = True
        else:
            self._st = False
            snapshot = resolve_local_hf_snapshot(model_name)
            if snapshot:
                self._resolved_path = snapshot
                logger.info("Embedding 模型使用本地 HF 快照: %s (无网络请求)", snapshot)

    def assert_dimension(self, expected: int | None = None) -> int:
        """断言实际维度 == 期望维度，返回实际维度。

        旧项目的教训（HANDOFF §9.2）：代码默认 bge-m3(1024) 而 Qdrant 集合是 512，
        应用层直接 400 而**无人发现**。所以这里不返回布尔值、不做告警，
        而是**抛错**——维度不一致时任何后续检索结果都无意义。
        """
        want = expected if expected is not None else self._expected_dimension
        actual = self.dimension
        if want is not None and actual != want:
            raise RuntimeError(
                f"Embedding 维度不一致：模型 {self._model_name!r} 实际 {actual} 维，"
                f"期望 {want} 维。请检查 EMBEDDING_DIMENSION 与 Qdrant 集合维度"
                f"（不匹配时检索结果无意义，因此直接失败而不是继续跑）。"
            )
        return actual

    def _get_model(self):
        """按需加载（lru_cache 缓存），首次加载后执行维度断言。"""
        if self._st:
            model = _load_st_model(
                self._resolved_path or self._model_path, self._device, self._fp16
            )
        else:
            model = _load_hf_model(self._resolved_path or self._model_path, self._device)
        if not self._checked:
            self._checked = True
            if self._expected_dimension is not None:
                self.assert_dimension(self._expected_dimension)
                logger.info(
                    "Embedding 维度断言通过: %s == %s", self._model_name, self._expected_dimension
                )
        return model

    def encode_sync(self, text: str) -> List[float]:
        """同步编码单条文本，返回归一化向量。"""
        model = self._get_model()
        if self._st:
            return model.encode(text, normalize_embeddings=True).tolist()
        return model.get_text_embedding(text)

    def encode_batch_sync(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """同步批量编码，返回向量列表。"""
        if not texts:
            return []
        model = self._get_model()
        if self._st:
            embs = model.encode(texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
            return [e.tolist() for e in embs]
        return [model.get_text_embedding(t) for t in texts]

    async def encode(self, text: str) -> List[float]:
        """异步编码单条文本（线程池包装同步实现）。"""
        return await asyncio.to_thread(self.encode_sync, text)

    async def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """异步批量编码（线程池包装同步实现）。"""
        return await asyncio.to_thread(self.encode_batch_sync, texts, batch_size)

    @property
    def dimension(self) -> int:
        """返回 embedding 维度。"""
        model = self._get_model()
        if self._st:
            return model.get_embedding_dimension()
        try:
            return model._model.config.hidden_size
        except Exception:
            test_emb = model.get_text_embedding("test")
            return len(test_emb)

    @property
    def is_sentence_transformers(self) -> bool:
        return self._st


def get_llamaindex_embedding():
    """返回适配 llama-index BaseEmbedding 接口的 embedding 对象。"""
    from llama_index.core.embeddings import BaseEmbedding

    class _LlamaIndexAdapter(BaseEmbedding):
        """把 EmbeddingModel 适配为 llama-index BaseEmbedding。"""

        def __init__(self, model: EmbeddingModel, **kwargs: Any) -> None:
            super().__init__(model_name=model._model_name, embed_batch_size=32, **kwargs)
            self._model = model

        def _get_text_embedding(self, text: str) -> List[float]:
            return self._model.encode_sync(text)

        def _get_query_embedding(self, query: str) -> List[float]:
            return self._model.encode_sync(query)

        async def _aget_text_embedding(self, text: str) -> List[float]:
            return await self._model.encode(text)

        async def _aget_query_embedding(self, query: str) -> List[float]:
            return await self._model.encode(query)

    return _LlamaIndexAdapter(get_embedding_model())


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """按 settings 构造并缓存全局 EmbeddingModel（lru_cache）。"""
    settings = get_settings()
    device = resolve_inference_device(settings)
    model_name = settings.embedding_model_name
    model_path = settings.embedding_model_path
    fp16 = bool(getattr(settings, "embedding_fp16", False)) and device == "cuda"
    log_device_context(f"Embedding: {model_name}", device)
    return EmbeddingModel(
        model_name=model_name,
        model_path=model_path,
        device=device,
        fp16=fp16,
    )


async def get_text_embedding(text: str) -> List[float]:
    """异步获取单条文本的 embedding。"""
    model = get_embedding_model()
    return await model.encode(text)


async def get_text_embeddings_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """异步批量获取文本 embedding。"""
    model = get_embedding_model()
    return await model.encode_batch(texts, batch_size=batch_size)


def get_text_embedding_sync(text: str) -> List[float]:
    """同步获取单条文本的 embedding。"""
    return get_embedding_model().encode_sync(text)
