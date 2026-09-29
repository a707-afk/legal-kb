# -*- coding: utf-8 -*-
"""法律语料导入：npc 官方 docx + 元数据 → 带时效 front-matter 的 Markdown（D1）。

与旧脚本（`eval/legacy-v3/scripts/ingest_legal_corpus.py`）的三个根本差别
--------------------------------------------------------------------
1. **全部版本入库**。旧脚本按"同名只保留最新"删掉了 431 个历史版本——**那正是本项目的题源**。
   本脚本以 `bbbs`（官网版本号，全局唯一 2,466 个）为版本标识，一条不丢。
2. **写入真正的时效元数据**。旧脚本读的字段名是 `status`，官方叫 `status_code`
   → 2,466 条时效元数据全部落空，front-matter 硬编码成 `current`，**整个能力变成空话**。
   本脚本**对字段做断言**（缺 `status_code` 键直接报错），不给默认值兜底。
3. **读 `.doc`**。语料里有 63 个 OLE2 旧版 `.doc`（司法解释一栏占 58 个），`python-docx` 读不了。
   本脚本优先读 `files/*.docx`，缺失时回退读 `_doc_converted/<category>/*.docx`
   （由 `scripts/convert_doc_to_docx.py` 用 Word COM 一次性转换）。

输出
----
    data/docs/statute/*.md           法律 + 行政法规
    data/docs/interpretation/*.md    司法解释
    data/docs/faq/*.md               LawBench 1-1（强同源，见报告中的风险提示）
    reports/ingest_npc_report.json   统计 + 校验结果

用法
----
    python scripts/ingest_npc.py --dry-run     # 只统计不写盘
    python scripts/ingest_npc.py               # 全量导入（先清空 data/docs 下三个子目录）
    python scripts/ingest_npc.py --no-clean    # 增量导入（保留已有文件）
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "npc"
CONVERTED = RAW / "_doc_converted"
LAWBENCH_1_1 = ROOT / "data" / "raw" / "lawbench" / "1-1.json"
DOCS_OUT = ROOT / "data" / "docs"
REPORT_PATH = ROOT / "reports" / "ingest_npc_report.json"

# ── status_code 语义映射（HANDOFF §4.1 已交叉验证）──────────────────
STATUS_MAP: dict[int, tuple[str, str]] = {
    3: ("current", "现行有效"),
    2: ("superseded", "已被修订"),
    1: ("repealed", "已废止"),
    4: ("pending", "尚未生效"),
    -1: ("unknown", "未知"),
}
STATUS_UNLABELED = ("unlabeled", "未标注")  # status_code is None

# 元数据必须存在的键 —— 少一个就说明上游格式变了，必须报错而不是兜底
REQUIRED_META_KEYS = (
    "bbbs", "title", "category", "status_code", "effective_date",
    "publish_date", "law_nature", "downloaded_file",
)

CATEGORIES = ("法律", "行政法规", "司法解释")
DOMAIN_OF = {"法律": "statute", "行政法规": "statute", "司法解释": "interpretation"}
SOURCE_TYPE_OF = {"法律": "statute", "行政法规": "statute", "司法解释": "interpretation"}

MIN_BODY_CHARS = 100  # 正文 < 100 字视为无正文占位条目，剔除

CN_NUM = "一二三四五六七八九十百千零〇0-9"
RE_BIAN = re.compile(rf"^第[{CN_NUM}]+编")
RE_ZHANG = re.compile(rf"^第[{CN_NUM}]+章")
RE_JIE = re.compile(rf"^第[{CN_NUM}]+节")
RE_TIAO = re.compile(rf"^第[{CN_NUM}]+条")
# 司法解释常用"一、一般规定"这类中文数字序号作节
RE_SECTION_CN = re.compile(rf"^[{CN_NUM}]+、")
# 附件：`附件一：` / `附件2` —— 一份文件含多个附件时，各附件**编号会重新从第一条开始**，
# 若不提升为标题，两个附件的"第一章 > 第二条"会撞成同一个 heading_path
# （实测 `重新组建仲裁机构方案` 因此产生同文档内重复节点，也让条号级引用校验无法定位）
RE_ATTACHMENT = re.compile(rf"^附\s*件\s*[{CN_NUM}]+\s*[：:]?\s*$|^附\s*件\s*[{CN_NUM}]+\s*[：:]")
RE_TOC = re.compile(r"^目\s*录$")
RE_FULLWIDTH_SPACE = re.compile(r"[　\s]+")

# LawBench 部门法前缀（用于从问句中剥出法名）
LAWBENCH_DEPARTMENTS = (
    "诉讼与非诉讼程序法", "宪法相关法", "民法商法", "行政法", "经济法", "社会法", "刑法",
)


# ══════════════════════════════════════════════════════════════════
# 读取
# ══════════════════════════════════════════════════════════════════

def load_metadata(category: str) -> list[dict]:
    path = RAW / category / "metadata.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"元数据不存在: {path}")
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSON 解析失败: {exc}") from exc
    return out


def assert_metadata_fields(records: list[dict], category: str) -> None:
    """字段断言 —— 防止旧项目"读错字段名导致元数据全落空"的事故重演。

    关键：`status_code` 必须**存在**（值可以是 None），不存在即视为上游格式变更。
    """
    if not records:
        raise ValueError(f"{category}: 元数据为空")
    for i, rec in enumerate(records):
        missing = [k for k in REQUIRED_META_KEYS if k not in rec]
        if missing:
            raise ValueError(
                f"{category}: 第 {i + 1} 条缺字段 {missing}。"
                f"若上游把字段改名（如 status_code -> status），必须同步改本脚本，"
                f"**不要用默认值兜底**（旧项目正是在这里丢掉了 2,466 条时效元数据）。"
            )
        sc = rec["status_code"]
        if sc is not None and sc not in STATUS_MAP:
            raise ValueError(f"{category}: 第 {i + 1} 条 status_code={sc!r} 不在已知语义内")


def resolve_source_file(rec: dict, category: str) -> Path | None:
    """定位正文文件。

    优先级：`files/<name>.docx` → `_doc_converted/<category>/<stem>.docx`（`.doc` 的转换产物）。
    ⚠️ 不能直接返回 `files/*.doc`——`python-docx` 读不了 OLE2 格式，
    否则 63 篇（司法解释占 58 篇）会以"读取失败"的名义静默丢失。
    """
    rel = (rec.get("downloaded_file") or "").replace("\\", "/")
    if not rel:
        return None
    name = Path(rel).name
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()

    if suffix != ".doc":
        direct = RAW / category / "files" / name
        if direct.is_file():
            return direct
    else:
        # .doc 优先找转换产物，找不到再退回原文件（会走"读取失败"分支并计入报告）
        converted = CONVERTED / category / f"{stem}.docx"
        if converted.is_file():
            return converted

    for candidate in (RAW / category / "files" / name, CONVERTED / category / f"{stem}.docx"):
        if candidate.is_file():
            return candidate
    return None


def docx_paragraphs(path: Path) -> list[str]:
    """读 docx 段落文本。

    先走 `python-docx`；失败时回退直接解析 `word/document.xml`
    （Word COM 转换出的个别文件缺 `officeDocument` 关系，python-docx 会拒绝打开，
    但正文 XML 本身是完整的）。
    """
    try:
        from docx import Document

        doc = Document(str(path))
        return [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    except Exception:
        return _docx_paragraphs_from_xml(path)


def _docx_paragraphs_from_xml(path: Path) -> list[str]:
    """直接解析 docx 包内的 word/document.xml，按 <w:p> 聚合 <w:t>。"""
    import zipfile
    import xml.etree.ElementTree as ET

    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    out: list[str] = []
    for p in root.iter(f"{W}p"):
        text = "".join(t.text or "" for t in p.iter(f"{W}t"))
        text = text.strip()
        if text:
            out.append(text)
    if not out:
        raise ValueError(f"正文为空（XML 回退解析）: {path.name}")
    return out


# ══════════════════════════════════════════════════════════════════
# 正文渲染
# ══════════════════════════════════════════════════════════════════

def _norm(s: str) -> str:
    """归一化：去掉全角/半角空白。"""
    return RE_FULLWIDTH_SPACE.sub("", s or "")


# 标题比对用的宽松归一化：连标点一起去掉。
# 官方 docx 里的标题常与元数据标题**标点不一致**（实测 docx 写
# `最高人民法院最高人民检察院关于...批复`，元数据写 `最高人民法院、最高人民检察院关于...批复`），
# 严格比对会导致标题段没被去掉、在正文里重复出现一次。
RE_PUNCT = re.compile(r"[、，。：；·—－\-《》〈〉（）()【】\[\]“”\"'’‘]")
def _norm_loose(s: str) -> str:
    return RE_PUNCT.sub("", _norm(s))


def _strip_leading_title(paras: list[str], title: str) -> list[str]:
    """移除开头与 `title` 重复的标题段（正文标题由渲染层统一写成 `# title`）。

    不能只从下标 0 开始比对：官方 docx 的司法解释常带"公告"抬头
    （`中华人民共和国最高人民检察院` / `公　　告` / 公布说明 / 落款 / 日期），
    **真标题在第 5 段左右**。所以做法是：在开头窗口内找一个**连续段**，
    其归一化拼接正好等于归一化标题，移除该段、保留其余（公告抬头与通过说明都有信息量）。
    """
    title_norm = _norm_loose(title)
    if not title_norm:
        return paras
    window = min(len(paras), 15)
    for i in range(window):
        acc = ""
        for j in range(i, min(i + 5, len(paras))):
            acc += _norm_loose(paras[j])
            if acc == title_norm:
                return paras[:i] + paras[j + 1:]
            if not title_norm.startswith(acc):
                break
    return paras


def _looks_like_body_start(paras: list[str], k: int) -> bool:
    """判断 `paras[k]` 之后是否紧跟实质内容（条文或长段落）。

    用来区分"目录里的条目"与"正文里的同名标题"：目录条目后面还是目录条目，
    正文标题后面会很快出现 `第X条` 或长段落。
    """
    for t in paras[k + 1: k + 6]:
        n = _norm(t)
        if not n:
            continue
        if RE_TIAO.match(n) or len(n) > 30:
            return True
    return False


def _body_start_after_toc(paras: list[str], start: int) -> int:
    """跳过"目录"块，返回正文起始下标。

    判据（三步，逐级回退）
    --------------------
    1. 定位"目录"标记，收集其后的**短行连续段**作为目录条目。
    2. 取目录中**只出现一次**的首个条目作签名（`第一章　总　则` 这类），
       在窗口内寻找它的下一次出现，且该位置**后面紧跟条文/长段落**
       （`_looks_like_body_start`）——这就是正文起点。
    3. 回退：任一"重复出现的条目"位置。

    为什么需要第 2 步的"后面紧跟实质内容"判据：目录里**不同章节会有同名小节**
    （实测 `人民检察院公益诉讼办案规则` 的 `第一节　立案与调查` 在第三章、第四章
    各出现一次），只按"首个重复条目"会把正文起点定在目录内部，
    导致目录后半段泄漏进正文。
    """
    toc_idx = None
    for i in range(start, len(paras)):
        if RE_TOC.match(_norm(paras[i])):
            toc_idx = i
            break
    if toc_idx is None:
        return start

    window_end = min(len(paras), toc_idx + 1 + 200)
    entries: list[tuple[int, str]] = []
    j = toc_idx + 1
    while j < window_end:
        n = _norm(paras[j])
        if not n:
            j += 1
            continue
        if RE_TIAO.match(n) or len(n) > 30:
            break
        entries.append((j, n))
        j += 1

    if not entries:
        return j

    # 第 2 步：以**目录的第一个条目**为签名（如 `第一章　总　则`），
    # 在窗口内找它的**最后一次**出现——正文里的同名标题必然在目录之后。
    # 取"最后一次"而不是"第一次"：正文首章标题也可能被短行收集吞进来，
    # 取第一次会落在目录内部（实测 `人民检察院公益诉讼办案规则` 曾因此跳过整个第一章）。
    sig = entries[0][1]
    candidates = [k for k in range(entries[0][0] + 1, window_end) if _norm(paras[k]) == sig]
    if candidates:
        body_like = [k for k in candidates if _looks_like_body_start(paras, k)]
        return body_like[-1] if body_like else candidates[-1]

    # 第 3 步：签名在正文里不再出现 → 目录块就是那段短行连续段，正文从其末尾开始
    return j


def toc_index(paras: list[str], start: int = 0) -> int | None:
    """返回"目录"标记段的下标；没有则 None。"""
    for i in range(start, len(paras)):
        if RE_TOC.match(_norm(paras[i])):
            return i
    return None


def render_statute(paras: list[str], title: str) -> tuple[str, bool]:
    """法规正文 → (带标题层级的 Markdown, 是否剥离了目录块)。

    层级自适应：有「编」时 编=##/章=###/节=####；无「编」时整体上提一级
    （章=##/节=###），避免出现"跳级"的 heading_path（D2 切块按标题路径生成，
    跳级会让 `header_path` 出现空层）。
    """
    paras = _strip_leading_title(paras, title)
    had_toc = toc_index(paras, 0) is not None
    start = _body_start_after_toc(paras, 0)
    rest = paras[start:]
    has_bian = any(RE_BIAN.match(t) for t in rest)
    h_bian, h_zhang, h_jie = ("##", "###", "####") if has_bian else ("", "##", "###")

    buf: list[str] = []
    for t in rest:
        if has_bian and RE_BIAN.match(t):
            buf.append(f"{h_bian} {t}")
        elif RE_ZHANG.match(t):
            buf.append(f"{h_zhang} {t}")
        elif RE_JIE.match(t):
            buf.append(f"{h_jie} {t}")
        elif RE_TIAO.match(t):
            head, _, tail = t.partition("　")
            if not tail:
                head, _, tail = t.partition(" ")
            buf.append(f"**{head}** {tail.strip()}")
        elif RE_SECTION_CN.match(t) and len(t) <= 24:
            # 司法解释的"一、一般规定"式小节
            buf.append(f"{h_jie} {t}")
        elif RE_ATTACHMENT.match(t):
            # 附件是**顶层划分**，必须用 `#`：若与 `## 第一章` 同级，
            # 切块时栈会把"附件二"弹掉，两个附件的"第一章 > 第二条"仍会撞车
            buf.append(f"# {t}")
        else:
            buf.append(t)

    # 段落间单空行，压掉连续空行
    out: list[str] = []
    for line in buf:
        if not line.strip():
            continue
        if out and (line.startswith("#") or out[-1].startswith("#")):
            out.append("")
        out.append(line)
    return "\n\n".join(out).strip(), had_toc


# ══════════════════════════════════════════════════════════════════
# front-matter
# ══════════════════════════════════════════════════════════════════

def slugify(title: str, version: str) -> str:
    """标题 + 完整版本号。

    ⚠️ **必须用完整 bbbs**：bbbs 是"抓取批次前缀 + 内容摘要"结构，
    实测前 8 位只有 70 个唯一值（`ff808181` 一项就占 554 条）、前 12 位仍有 94 组冲突，
    只有**后 12 位或全量**才全局唯一（2,466/2,466）。
    用短前缀会静默覆盖——旧项目"评测集同名覆盖三次"的同类事故。
    最长路径实测 143 字符，远低于 Windows 260 上限。
    """
    bad = '<>:"/\\|?*\n\r\t'
    t = "".join(c for c in title if c not in bad).strip()
    return f"{t[:70]}_{version}"


def build_front_matter(
    *,
    domain: str,
    subdomain: str,
    source_type: str,
    title: str,
    version: str,
    status_code: int | None,
    issuer: str,
    publish_date: str,
    effective_date: str,
    law_nature: str,
    source_category: str,
    source_origin: str,
) -> str:
    status, status_label = STATUS_MAP.get(status_code, STATUS_UNLABELED) if status_code is not None \
        else STATUS_UNLABELED
    sc_repr = "null" if status_code is None else str(status_code)
    q = lambda v: str(v or "").replace('"', "'").replace("\n", " ").strip()  # noqa: E731
    return (
        "---\n"
        f"domain: {domain}\n"
        f'subdomain: "{q(subdomain)}"\n'
        f'source_type: "{q(source_type)}"\n'
        f'title: "{q(title)}"\n'
        "security_level: public\n"
        "tenant_id: corp-default\n"
        "audience: legal,public\n"
        f"status_code: {sc_repr}\n"
        f'status: "{status}"\n'
        f'status_label: "{status_label}"\n'
        f'version: "{q(version)}"\n'
        f'law_nature: "{q(law_nature)}"\n'
        f'source_category: "{q(source_category)}"\n'
        f'issuer: "{q(issuer)}"\n'
        f'publish_date: "{q(publish_date)}"\n'
        f'effective_date: "{q(effective_date)}"\n'
        f'source_origin: "{q(source_origin)}"\n'
        "---\n\n"
    )


# ══════════════════════════════════════════════════════════════════
# 三路导入
# ══════════════════════════════════════════════════════════════════

def ingest_npc_category(category: str, stats: dict, dry_run: bool) -> None:
    records = load_metadata(category)
    assert_metadata_fields(records, category)

    domain = DOMAIN_OF[category]
    out_dir = DOCS_OUT / domain
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_short = 0
    missing_file: list[str] = []
    slugs: Counter[str] = Counter()
    toc_stripped: list[str] = []

    for rec in records:
        title = rec["title"]
        bbbs = rec["bbbs"]
        src = resolve_source_file(rec, category)
        if src is None:
            missing_file.append(rec.get("downloaded_file", ""))
            continue

        try:
            paras = docx_paragraphs(src)
        except Exception as exc:  # 单篇失败不中断整体
            missing_file.append(f"{rec.get('downloaded_file','')} (读取失败: {exc})")
            continue

        body, had_toc = render_statute(paras, title)
        if len(_norm(body)) < MIN_BODY_CHARS:
            skipped_short += 1
            continue
        if had_toc:
            toc_stripped.append(bbbs)

        slug = slugify(title, bbbs)
        slugs[slug] += 1
        fm = build_front_matter(
            domain=domain,
            subdomain=category,
            source_type=SOURCE_TYPE_OF[category],
            title=title,
            version=bbbs,
            status_code=rec.get("status_code"),
            issuer=rec.get("issuer", ""),
            publish_date=rec.get("publish_date", ""),
            effective_date=rec.get("effective_date", ""),
            law_nature=rec.get("law_nature", ""),
            source_category=category,
            source_origin=f"npc/{category}/files/{Path(rec.get('downloaded_file','')).name}",
        )
        if not dry_run:
            (out_dir / f"{slug}.md").write_text(fm + f"# {title}\n\n" + body + "\n", encoding="utf-8")
        kept += 1

    dup_slugs = {k: v for k, v in slugs.items() if v > 1}
    if dup_slugs:
        # 静默覆盖是旧项目的头号事故（评测集同名被覆盖三次导致结果不可复现）。
        # 语料层同样不能容忍：宁可报错，不可丢文件。
        raise RuntimeError(
            f"{category}: 输出文件名冲突 {len(dup_slugs)} 组，会静默覆盖。"
            f"示例: {list(dup_slugs.items())[:3]}"
        )
    stats["npc"][category] = {
        "metadata_records": len(records),
        "kept": kept,
        "skipped_body_too_short": skipped_short,
        "missing_or_unreadable_file": len(missing_file),
        "missing_samples": missing_file[:5],
        "toc_stripped_count": len(toc_stripped),
        "toc_stripped_bbbs": toc_stripped,
        "duplicate_slugs": dup_slugs,
        "status_code_dist": dict(sorted(
            Counter(r.get("status_code") for r in records).items(),
            key=lambda x: (x[0] is None, x[0]),
        )),
        "unique_titles": len({r["title"] for r in records}),
        "multi_version_titles": sum(
            1 for _, v in Counter(r["title"] for r in records).items() if v > 1
        ),
    }


def extract_lawbench_subdomain(question: str) -> str:
    """从 LawBench 问句里剥出法名。

    问句形如 `民法商法农民专业合作社法第三十三条的内容是什么？`
    → 先剥部门法前缀（"民法商法"），剩余部分到"第"之前即法名（"农民专业合作社法"）。

    旧代码是 `q.split("法")[0][:6]` —— 对上面这句会得到 **"民"**，把子域打成噪音。
    """
    q = (question or "").strip()
    m = re.match(r"^(.*?)第", q)
    head = m.group(1) if m else q
    for dept in LAWBENCH_DEPARTMENTS:
        if head.startswith(dept):
            head = head[len(dept):]
            break
    # 宪法修正案：`宪法修正案2004年第二十三条...`
    m2 = re.match(r"^(宪法修正案\d{4}年)", head)
    if m2:
        return m2.group(1)
    return head or "综合"


def ingest_lawbench(stats: dict, dry_run: bool) -> None:
    if not LAWBENCH_1_1.is_file():
        stats["faq"] = {"error": f"未找到 {LAWBENCH_1_1}"}
        return
    items = json.loads(LAWBENCH_1_1.read_text(encoding="utf-8"))
    out_dir = DOCS_OUT / "faq"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_empty = 0
    skipped_duplicate = 0
    subdomains: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str]] = set()
    for i, it in enumerate(items):
        q = (it.get("question") or "").strip()
        a = re.sub(r"^答案[:：]", "", (it.get("answer") or "").strip()).strip()
        if not q or not a:
            skipped_empty += 1
            continue
        # 源数据含重复的 (问题, 答案) 对（实测 2 组）→ 会产生字节级重复节点，
        # 违反"重复节点 = 0"的验收，必须在入库层去掉
        pair = (q, a)
        if pair in seen_pairs:
            skipped_duplicate += 1
            continue
        seen_pairs.add(pair)
        sub = extract_lawbench_subdomain(q)
        subdomains[sub] += 1
        fm = build_front_matter(
            domain="faq",
            subdomain=sub,
            source_type="faq",
            title=q.rstrip("？?"),
            version=f"lb1-1-{i:04d}",
            status_code=None,  # LawBench 无时效元数据，如实标"未标注"，不猜
            issuer="",
            publish_date="",
            effective_date="",
            law_nature="",
            source_category="LawBench",
            source_origin="LawBench/zero_shot/1-1.json",
        )
        if not dry_run:
            (out_dir / f"{slugify(q.rstrip('？?'), f'lb1-1-{i:04d}')}.md").write_text(
                fm + f"# {q}\n\n{a}\n", encoding="utf-8"
            )
        kept += 1

    stats["faq"] = {
        "source": "LawBench/zero_shot/1-1.json",
        "items": len(items),
        "kept": kept,
        "skipped_empty": skipped_empty,
        "skipped_duplicate": skipped_duplicate,
        "distinct_subdomains": len(subdomains),
        "top_subdomains": subdomains.most_common(10),
        "risk": "强同源：问句形如「《X法》第Y条的内容是什么？」，与 A 类法条精确题同构。"
                "按 docs/02 D-08，A 类只能作回归基线，不得当作能力证据。",
    }


# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="npc 官方语料 -> data/docs（全部版本 + 时效元数据）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    ap.add_argument("--no-clean", action="store_true", help="不清空目标目录（增量导入）")
    ap.add_argument("--skip-faq", action="store_true", help="跳过 LawBench FAQ")
    args = ap.parse_args()

    if not args.dry_run and not args.no_clean:
        for sub in ("statute", "interpretation", "faq"):
            d = DOCS_OUT / sub
            if d.is_dir():
                shutil.rmtree(d)
                print(f"已清空 {d.relative_to(ROOT)}")

    stats: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "min_body_chars": MIN_BODY_CHARS,
        "npc": {},
        "case": {"included": False, "reason": "D-15：PII + 金标在库内导致评测失真，第一轮不纳入"},
    }

    for cat in CATEGORIES:
        ingest_npc_category(cat, stats, args.dry_run)
        row = stats["npc"][cat]
        print(f"[{cat}] 元数据 {row['metadata_records']} → 入库 {row['kept']}"
              f"  (过短剔除 {row['skipped_body_too_short']}, 文件缺失 {row['missing_or_unreadable_file']})")

    if not args.skip_faq:
        ingest_lawbench(stats, args.dry_run)
        print(f"[FAQ] 入库 {stats['faq'].get('kept', 0)}")

    # ── 汇总与验收 ──
    total_docs = sum(v["kept"] for v in stats["npc"].values()) + stats.get("faq", {}).get("kept", 0)
    stats["summary"] = {
        "docs_total": total_docs,
        "multi_version_titles": sum(v["multi_version_titles"] for v in stats["npc"].values()),
        "unique_titles": sum(v["unique_titles"] for v in stats["npc"].values()),
        "metadata_total": sum(v["metadata_records"] for v in stats["npc"].values()),
    }

    if not args.dry_run:
        actual = len(list(DOCS_OUT.rglob("*.md")))
        stats["summary"]["docs_on_disk"] = actual
        stats["summary"]["acceptance_ge_2400"] = actual >= 2400

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(json.dumps(stats["summary"], ensure_ascii=False, indent=1))
    print(f"报告: {REPORT_PATH.relative_to(ROOT)}")
    if not args.dry_run and not stats["summary"].get("acceptance_ge_2400", True):
        print("⚠️ 验收未通过：docs 总数 < 2,400", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
