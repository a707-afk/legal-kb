"""Multi-strategy chunking: hierarchical markdown → recursive sentence split → parent-child structure.

Strategies (configurable via chunk_strategy):
- `hierarchical_recursive` (default): Heading-aware split + recursive sentence split +
  overlap + parent-child metadata.
- `markdown_heading_overlap`: Heading split + SentenceSplitter (legacy).
- `heading_only`: Heading split only (no further subdivision).

All strategies preserve tenant_id, access-control metadata, and heading-path tracking.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode

from src.config_LEGACY_REFERENCE import Settings

_DEFAULT_TENANT_ID = "corp-default"
_ACCESS_METADATA_KEYS = (
    "domain",
    "subdomain",
    "security_level",
    "audience",
    "tenant_id",
    "owner",
    "status",
    "version",
)

logger = logging.getLogger(__name__)


# ── Metadata helpers (preserved from original) ────────────────────

def _ensure_access_metadata(metadata: dict) -> dict:
    out = dict(metadata)
    if not str(out.get("tenant_id") or "").strip():
        out["tenant_id"] = _DEFAULT_TENANT_ID
    return out


def _rel_path_under_docs(file_path: str, docs_dir: Path) -> str:
    if not (file_path or "").strip():
        return ""
    dd = docs_dir.resolve()
    raw = file_path.strip()
    try:
        p = Path(raw).resolve()
        return str(p.relative_to(dd))
    except ValueError:
        pass
    p2 = Path(raw)
    if not p2.is_absolute():
        cand = (dd / raw).resolve()
        try:
            return str(cand.relative_to(dd))
        except ValueError:
            pass
    name = p2.name
    if name:
        matches = [m for m in dd.rglob(name) if m.is_file() and m.suffix.lower() == ".md"]
        if len(matches) == 1:
            return str(matches[0].relative_to(dd))
        if len(matches) > 1:
            matches.sort(key=lambda x: str(x))
            logger.warning(
                "同 basename 多文件命中，取字典序第一个: %s (candidates=%s)",
                name,
                [str(m.relative_to(dd)) for m in matches[:5]],
            )
            return str(matches[0].relative_to(dd))
    return name


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    text = raw.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, raw
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    end_idx: int | None = None
    for i in range(1, len(lines)):
        stripped = lines[i].strip()
        if stripped == "---" or stripped.startswith("---"):
            end_idx = i
            break
    if end_idx is None:
        return {}, raw
    meta: dict[str, str] = {}
    for line in lines[1:end_idx]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            meta[key] = value

    # `status_code` 是时效治理（D-02/D-06）的判据字段，必须保持**类型正确**。
    # front-matter 是纯文本解析，`status_code: null`（未标注）会变成字符串 "null"、
    # `status_code: 3` 会变成字符串 "3"——下游 `int()` / `== 3` 都会静默出错。
    # 只对这一个键做强制转换，避免误伤 `version`（bbbs 可能全为数字，转 int 会丢前导零）。
    if "status_code" in meta:
        raw_sc = str(meta["status_code"]).strip().lower()
        meta["status_code"] = None if raw_sc in ("", "null", "none", "~") else int(raw_sc)

    body = "\n".join(lines[end_idx + 1:]).lstrip()
    return meta, body


def _normalize_metadata_value(value) -> str | int | float | bool | None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v) for v in value)
    return str(value)


def _normalize_node_metadata(nodes: list[BaseNode]) -> None:
    for node in nodes:
        raw = dict(node.metadata or {})
        node.metadata = _ensure_access_metadata(
            {str(k): _normalize_metadata_value(v) for k, v in raw.items()}
        )


def _doc_metadata_by_path(documents: list[Document], docs_dir: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for doc in documents:
        fp = doc.metadata.get("file_path") or doc.metadata.get("file_name") or ""
        if fp:
            rel = _rel_path_under_docs(fp, docs_dir).replace("\\", "/")
            doc.metadata["file_name"] = Path(rel).name
            doc.metadata["file_path"] = rel
        else:
            rel = ""
            doc.metadata.setdefault("file_name", "unknown")
            doc.metadata.setdefault("file_path", "")
        base = _ensure_access_metadata(dict(doc.metadata or {}))
        if rel:
            lookup[rel] = base
    return lookup


def _inherit_doc_metadata(nodes: list[BaseNode], doc_lookup: dict[str, dict[str, str]]) -> None:
    for node in nodes:
        raw = dict(node.metadata or {})
        fp = str(raw.get("file_path") or "").replace("\\", "/")
        inherited = doc_lookup.get(fp, {})
        merged = {**inherited, **raw}
        node.metadata = merged


def load_documents(docs_dir: Path) -> list[Document]:
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"docs_dir 不存在: {docs_dir}")
    documents: list[Document] = []
    for path in sorted(docs_dir.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(raw)
        rel_path = _rel_path_under_docs(str(path), docs_dir)
        metadata = _ensure_access_metadata({
            **frontmatter,
            "file_name": path.name,
            "file_path": rel_path,
            "source_path": rel_path,
            "doc_group": Path(rel_path).parts[0] if Path(rel_path).parts else "",
        })
        documents.append(Document(text=body, metadata=metadata))
    return documents


# ── New: Heading-aware chunking with heading-path metadata ────────

def _extract_heading_path(heading_node: BaseNode) -> str:
    """Return the heading-path string from a heading node's metadata."""
    return str(heading_node.metadata.get("header_path", "") or "").strip()


def _heading_level(heading_path: str) -> int:
    """Infer heading depth from the path separator count."""
    if not heading_path:
        return 0
    return heading_path.count(" > ") + 1


def _truncate_at_sentence_boundary(text: str, max_chars: int) -> int:
    """Find the best split point near max_chars, preferring sentence/line breaks."""
    if len(text) <= max_chars:
        return len(text)
    # Try to find the last sentence boundary within max_chars
    candidate = max_chars
    # Look for sentence-ending punctuation followed by space/newline
    for sep in ("。\n", "！\n", "？\n", ".\n", "\n\n", "。", "！", "？", ". ", ".\n", "\n"):
        idx = text.rfind(sep, 0, candidate)
        if idx != -1 and idx > candidate * 0.6:
            return idx + len(sep)
    # Fallback: line break
    idx = text.rfind("\n", 0, candidate)
    if idx != -1 and idx > candidate * 0.5:
        return idx + 1
    # Last resort: space boundary
    idx = text.rfind(" ", 0, candidate)
    if idx != -1 and idx > candidate * 0.4:
        return idx + 1
    return candidate


# ── Bounded recursive split（重写：数学上可证明的有界实现）──────────
#
# 设计不变量（对应有界性要求 a/b/c，对任意输入成立）：
#   记 raw_step = max(1, max_chars - overlap_chars)，step = max(raw_step, min(64, max_chars))。
#   a. 切点搜索只在窗口 [start + step, min(start + max_chars, len(text))] 内做
#      rfind；min_progress_floor = max(1, max_chars - overlap_chars) 的 100%
#      （≥ 要求的 40%），绝不从全文位置 0 回看（旧实现的 rfind(sep, 0, ·) 是
#      O(n²) 与跨块回看的根源）。
#   b. 每轮迭代 start 严格前进 step（≥1）个字符；且切点 cut ≥ start + step，
#      故相邻块覆盖区间 [start, cut) 与 [start+step, ·) 重叠或相接——不跳字、
#      不回退、必然终止。
#   c. 迭代次数恰为 ceil(len/step) ≤ len//max(1, max_chars-overlap) + 16；
#      块数 ≤ 迭代数（合并只会减少）。max_chars ≥ 64 时 step ≥ 64，块数
#      ≤ len//64 + 16。另设运行时硬护栏兜底（在上述数学保证下正常参数不可达）。
_SENTENCE_ENDERS = ("。", "！", "？", "；", "\n")


def _find_bounded_split(text: str, lo: int, hi: int) -> int:
    """在窗口 [lo, hi] 内找距 hi 最近的句界切点（切点落在句终符之后）。

    只在 [lo, hi] 内 rfind，绝不回看窗口之外；找不到句界则硬切返回 hi。
    返回值满足 lo < cut ≤ hi（hi ≤ len(text)）。
    """
    best = -1
    for ch in _SENTENCE_ENDERS:
        idx = text.rfind(ch, lo, hi)
        if idx >= 0 and idx + 1 > best:
            best = idx + 1
    return best if best >= 0 else hi


def _next_preview(text: str, cut: int, overlap_chars: int) -> str:
    """取下一块开头约 overlap_chars 的预览，尽量截断在句界/空白处。"""
    preview = text[cut:cut + overlap_chars]
    for sep in ("。", "\n", ".", " ", "！", "？"):
        j = preview.find(sep)
        if j != -1:
            return preview[:j + 1]
    return preview


def _recursive_split_text(
    text: str,
    max_chars: int,
    overlap_chars: int,
    min_chunk_chars: int = 50,
) -> list[dict]:
    """Bounded recursive sentence split with overlap.

    Returns list of {"text": str, "overlap_prev": str, "overlap_next": str}.

    语义与有界性（证明见上方不变量注释）：
    - 每块 text ≤ max_chars；相邻块重叠 ∈ [0, overlap_chars]（窗口内切到句界时
      ≈ overlap_chars）；
    - 小尾巴（< min_chunk_chars）并入前一块，合并后总长 ≤ 2×max_chars，否则独立成块；
    - 边界搜索只在 [start + step, start + max_chars] 窗口内，绝不回看全文 0 位；
    - 每轮 start 严格前进 step ≥ 1，迭代数与块数有确定硬上限。
    """
    if not text.strip():
        return []
    max_chars = max(1, int(max_chars))
    # overlap 钳制到 [0, max_chars-1]，保证 step ≥ 1（病态 overlap 不破坏有界性）
    overlap_chars = max(0, min(int(overlap_chars), max_chars - 1))
    min_chunk_chars = max(1, int(min_chunk_chars))
    if len(text) <= max_chars:
        return [{"text": text.strip(), "overlap_prev": "", "overlap_next": ""}]

    min_progress_floor = max(1, max_chars - overlap_chars)
    step = max(min_progress_floor, min(64, max_chars))
    max_iters = len(text) // max(1, max_chars - overlap_chars) + 16
    max_chunks = len(text) // 64 + 16
    merged_cap = max_chars * 2

    chunks: list[dict] = []
    start = 0
    iters = 0
    while start < len(text):
        iters += 1
        if iters > max_iters or len(chunks) >= max_chunks:
            # 硬护栏兜底（step ≥ 1 的数学保证下、正常参数不可达）：按 max_chars
            # 硬切收尾，保证终止、内容不丢、每块 ≤ max_chars。
            logger.warning(
                "_recursive_split_text 触发硬护栏（iters=%d, chunks=%d, len=%d）；"
                "剩余文本按 max_chars 硬切收尾",
                iters,
                len(chunks),
                len(text),
            )
            pos = start
            while pos < len(text):
                piece = text[pos:pos + max_chars].strip()
                if piece:
                    chunks.append({"text": piece, "overlap_prev": "", "overlap_next": ""})
                pos += max_chars
            break

        end_candidate = min(start + max_chars, len(text))
        # 窗口 [start+step, end_candidate]：绝不回看全文 0 位（要求 a）
        cut = _find_bounded_split(text, start + step, end_candidate)
        chunk_text = text[start:cut].strip()
        if chunk_text:
            if (
                len(chunk_text) < min_chunk_chars
                and chunks
                and len(chunks[-1]["text"]) + 1 + len(chunk_text) <= merged_cap
            ):
                # 小尾巴并入前一块（合并后 ≤ 2×max_chars，超出则独立成块）
                chunks[-1]["text"] += "\n" + chunk_text
            else:
                # Overlap with previous chunk
                overlap_prev = ""
                if chunks and overlap_chars > 0:
                    prev_text = chunks[-1]["text"]
                    overlap_prev = (
                        prev_text[-overlap_chars:] if len(prev_text) > overlap_chars else prev_text
                    )
                # Overlap with next segment (preview next ~overlap_chars)
                overlap_next = ""
                if overlap_chars > 0 and cut < len(text):
                    overlap_next = _next_preview(text, cut, overlap_chars)
                chunks.append({
                    "text": chunk_text,
                    "overlap_prev": overlap_prev,
                    "overlap_next": overlap_next,
                })
        # 要求 b：每轮严格前进 step（≥1）。由窗口下界保证 step ≤ cut - start
        # （cut = len(text) 时 start 直接越界退出），不跳字、不回退、必然终止。
        start += step
    return chunks


def _split_markdown_sections(text: str) -> list[tuple[str, str, int]]:
    """线性切节：每个标题只拥有它到下一个标题之间的正文（无嵌套复制）。

    返回 [(heading_path, section_text, level)]。嵌套层级只用于构造
    heading_path，不重复携带上层正文——避免 MarkdownNodeParser 嵌套
    语义导致的 O(深度×篇幅) 内容复制。
    """
    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")
    sections: list[tuple[str, str, int]] = []
    stack: list[tuple[int, str]] = []
    cur_path = ""
    cur_level = 0
    cur_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(cur_lines).strip()
        if body:
            sections.append((cur_path, body, cur_level))

    for line in text.split("\n"):
        m = heading_re.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur_path = " > ".join(t for _, t in stack)
            cur_level = level
            cur_lines = [line]
        else:
            cur_lines.append(line)
    flush()
    return sections


# ── 条级切分（BLUEPRINT 1.2：法条按"条"为最小单元）────────────────────

RE_TIAO_LINE = re.compile(
    r"^\*\*\s*(第[一二三四五六七八九十百千零〇0-9]+条)\s*\*\*"
)


def _split_legal_sections(text: str) -> list[tuple[str, str, int]]:
    """在 **markdown 标题** 与 **`**第X条**` 标记** 两处切节。

    为什么必须切到"条"（BLUEPRINT 1.2 的既定设计，`hierarchical_recursive` 之前没实现）：
    - **A 类法条精确题的判分口径是"文件命中 AND 条号命中"**，其中条号命中要求
      "top-1 片段的 `heading_path` 含该条，或片段正文以该条号开头"。
      若不按条切，一个片段里塞着好几条，两个条件都不成立 → **检索正确也会判失败**。
    - **D 类版本冲突题**要判"返回的是哪一版"。按条切后，同一部法的不同版本
      每个"条"是独立节点，各自带 `status_code`，版本判别才有落点。
    - 按条切后 `heading_path` 形如 `第一章 总则 > 第二条`，条号级引用校验（D-07）才有依据。

    长条（> chunk_chars）仍由调用方递归切分；短条保持整条不切。
    """
    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")
    sections: list[tuple[str, str, int]] = []
    stack: list[tuple[int, str]] = []
    cur_path = ""
    cur_level = 0
    cur_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(cur_lines).strip()
        if body:
            sections.append((cur_path, body, cur_level))

    for line in text.split("\n"):
        m = heading_re.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur_path = " > ".join(t for _, t in stack)
            cur_level = level
            cur_lines = [line]
            continue

        mt = RE_TIAO_LINE.match(line.strip())
        if mt:
            flush()
            tiao = mt.group(1)
            # 条挂在最近的标题之下；level 取"标题深度 + 1"，保证层级单调
            cur_path = " > ".join([t for _, t in stack] + [tiao]) if stack else tiao
            cur_level = len(stack) + 1
            cur_lines = [line]
            continue

        cur_lines.append(line)
    flush()
    return sections


RE_HEADING_ONLY = re.compile(r"^(#{1,6})\s+")


def _has_substantive_body(body: str, min_chars: int = 10) -> bool:
    """判断片段除去标题行后是否还有实质内容。

    纯标题节点（如 `## 第六章　强制措施` 后面直接跟 `### 第一节`）没有任何检索价值，
    却会**跨文档大量重复**（不同法规常有同名章节），是重复节点的主要来源之一。
    """
    rest = [
        ln for ln in (body or "").split("\n")
        if ln.strip() and not RE_HEADING_ONLY.match(ln.strip())
    ]
    return len(re.sub(r"\s+", "", "".join(rest))) >= min_chars


def _context_header(meta: dict, heading_path: str) -> str:
    """构造片段正文的上下文头：「《法名》 章 条」。

    为什么必须写进**正文**（而不是只放 metadata）：
    - BM25 是词项精确匹配、向量是正文语义，两路都**只看 text**；
    - 法条正文天然不含法名（《农业法》第一条正文通篇没有"农业法"三个字），
      于是"《中华人民共和国农业法》现在生效了吗"这类**按名查询**在两路都匹配不到目标法。
    实测后果：该法 217 个片段全部落选，BM25 top-30 里 29 条是 LawBench 短 FAQ
    （详见 `scripts/probe_retrieval.py` / `probe_bm25.py` 的复现）。

    头部只取法名 + 章/条尾段，避免把整个 heading_path 抄进来造成重复。
    """
    title = str(meta.get("title") or "").strip().strip("《》")
    parts = [p.strip() for p in (heading_path or "").split(">") if p.strip()]
    if parts and title and parts[0].strip("《》") == title:
        parts = parts[1:]
    tail = " ".join(parts)
    if title and tail:
        return f"《{title}》 {tail}"
    if title:
        return f"《{title}》"
    return tail


def _with_header(header: str, body: str) -> str:
    return f"{header}\n{body}" if header else body


def _build_hierarchical_nodes(
    documents: list[Document],
    settings: Settings,
    doc_lookup: dict[str, dict[str, str]],
) -> list[BaseNode]:
    """Build nodes: heading sections (linear, no duplication) + recursive
    sentence split for oversized sections, with parent-child metadata.

    每节正文只入库一次：小节 → 单节点（parent=child）；
    大节 → 父节点（正文截断到 2×chunk_chars，防止巨型节点撑爆内存/嵌入）
         + 子节点（递归句级切分，带 overlap）。
    """
    CHAR_PER_TOKEN = 2.0
    chunk_chars = int(settings.chunk_size_tokens * CHAR_PER_TOKEN)
    overlap_chars = int(settings.chunk_overlap_tokens * CHAR_PER_TOKEN)
    min_chunk_chars = max(50, chunk_chars // 8)
    parent_cap = chunk_chars * 2

    all_nodes: list[BaseNode] = []

    for doc in documents:
        doc_meta = dict(doc.metadata or {})
        fp = str(doc_meta.get("file_path") or "")
        inherited = doc_lookup.get(fp.replace("\\", "/"), {})
        merged_base = {**inherited, **doc_meta}

        text = doc.text or ""
        if not text.strip():
            continue
        heading_lines = len([ln for ln in text.split("\n") if ln.lstrip().startswith("#")])
        if heading_lines == 0:
            sections = [("", text, 0)]
        elif getattr(settings, "chunk_split_on_tiao", True):
            # BLUEPRINT 1.2：法条按"条"为最小单元（见 _split_legal_sections 的说明）
            sections = _split_legal_sections(text)
        else:
            sections = _split_markdown_sections(text)

        for heading_path, body, level in sections:
            if not body.strip():
                continue
            # 纯标题节点（后面直接跟下级标题）无检索价值，且跨文档大量重复 → 丢弃
            if not _has_substantive_body(body):
                continue
            base_meta = {
                **merged_base,
                "heading_path": heading_path,
                "heading_level": level,
            }
            header = _context_header(base_meta, heading_path)
            if len(body) <= chunk_chars:
                all_nodes.append(TextNode(
                    text=_with_header(header, body),
                    metadata={
                        **base_meta,
                        "is_parent_chunk": True,
                        "is_child_chunk": True,
                        "parent_heading_path": heading_path,
                    },
                ))
            else:
                parent_node = TextNode(
                    text=_with_header(header, body[:parent_cap]),
                    metadata={
                        **base_meta,
                        "is_parent_chunk": True,
                        "is_child_chunk": False,
                        "parent_heading_path": heading_path,
                        "chunk_count": 0,
                    },
                )
                all_nodes.append(parent_node)
                raw_chunks = _recursive_split_text(
                    body, chunk_chars, overlap_chars, min_chunk_chars
                )
                for i, rc in enumerate(raw_chunks):
                    if not rc["text"].strip():
                        continue
                    all_nodes.append(TextNode(
                        text=_with_header(header, rc["text"]),
                        metadata={
                            **base_meta,
                            "is_parent_chunk": False,
                            "is_child_chunk": True,
                            "parent_heading_path": heading_path,
                            "child_index": i,
                            "child_total": len(raw_chunks),
                            "overlap_prev": rc["overlap_prev"],
                            "overlap_next": rc["overlap_next"],
                        },
                    ))
                parent_node.metadata["chunk_count"] = len(raw_chunks)

    _normalize_node_metadata(all_nodes)
    return all_nodes


# ── Public entrypoint ─────────────────────────────────────────────

def build_nodes(documents: list[Document], settings: Settings) -> list[BaseNode]:
    """Build chunks from documents using the configured strategy.

    Supports strategies:
    - ``hierarchical_recursive`` (default): heading-aware + recursive sentence split
      + parent-child structure with overlap metadata.
    - ``markdown_heading_overlap``: heading split + SentenceSplitter (legacy).
    - ``heading_only``: heading split only.
    """
    if not documents:
        return []

    docs_dir = Path(settings.docs_dir).resolve()
    doc_lookup = _doc_metadata_by_path(documents, docs_dir)
    strategy = (settings.chunk_strategy or "hierarchical_recursive").strip().lower()

    if strategy == "heading_only":
        md_parser = MarkdownNodeParser.from_defaults(header_path_separator=" > ")
        nodes = md_parser.get_nodes_from_documents(documents)
        _inherit_doc_metadata(nodes, doc_lookup)
        _normalize_node_metadata(nodes)
        return nodes

    if strategy == "markdown_heading_overlap":
        md_parser = MarkdownNodeParser.from_defaults(header_path_separator=" > ")
        heading_nodes = md_parser.get_nodes_from_documents(documents)
        splitter = SentenceSplitter(
            chunk_size=settings.chunk_size_tokens,
            chunk_overlap=settings.chunk_overlap_tokens,
        )
        nodes = splitter(heading_nodes)
        _inherit_doc_metadata(nodes, doc_lookup)
        _normalize_node_metadata(nodes)
        return nodes

    if strategy == "hierarchical_recursive":
        return _build_hierarchical_nodes(documents, settings, doc_lookup)

    raise ValueError(
        f"未知 chunk_strategy={settings.chunk_strategy!r}；"
        "可选: hierarchical_recursive, markdown_heading_overlap, heading_only"
    )
