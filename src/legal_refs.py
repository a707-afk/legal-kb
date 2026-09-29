"""法条引用解析：中文数字归一化 + 条号/项号抽取。

为什么需要单独一个模块（D-06 的地基）
------------------------------------
法律语料的条号有两种写法，**必须归一化后才能比较**：

- 语料原文（官方 docx）：`第五百七十七条`、`第三十八条第（二）项`  ← 中文数字
- 模型生成答案：常常写成 `第577条`、`第584条`                    ← 阿拉伯数字

不做归一化会**双向出错**：
- 判分侧：把正确的引用判成"没命中"（我实测踩过——`sensenova-6.8-flash-lite`
  在买卖合同题上写了 `第577条` 与 `第五百七十七条` 两种写法，
  朴素正则把前者判为幻觉，条号精度从 1.0 掉到 0.781）
- 校验侧（D-07）：条号级引用校验会大量**误杀**合法引用

所以抽取器统一输出**阿拉伯数字**形式（`第577条` / `第2项`），比较前两侧都归一化。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ── 中文数字 ───────────────────────────────────────────────────────

_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}
_CN_CHARS = "".join(_CN_DIGIT) + "".join(_CN_UNIT) + "0-9０-９"


def cn2arabic(text: str) -> int | None:
    """中文数字 → 整数。支持 `十/十五/二十三/一百零五/五百七十七/一千二百三十四`。

    解析失败返回 ``None``（调用方决定是跳过还是报错，不猜）。
    """
    s = (text or "").strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    # 全角数字
    if all("\uff10" <= c <= "\uff19" for c in s):
        return int("".join(str(ord(c) - ord("\uff10")) for c in s))

    total = 0
    section = 0
    number = 0
    matched = False
    for ch in s:
        if ch in _CN_DIGIT:
            number = _CN_DIGIT[ch]
            matched = True
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            matched = True
            if unit == 10000:
                section = (section + (number or 1)) * unit
                total += section
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
        else:
            return None
    if not matched:
        return None
    return total + section + number


# ── 引用抽取 ───────────────────────────────────────────────────────

# 条：第X条；项：第X项 或 （X）；款：第X款
RE_ARTICLE = re.compile(rf"第\s*([{_CN_CHARS}]+)\s*条")
RE_ITEM_KUAN = re.compile(rf"第\s*([{_CN_CHARS}]+)\s*([项款])")
RE_PAREN_ITEM = re.compile(rf"[（(]\s*([{_CN_CHARS}]+)\s*[）)]")
RE_LAW_NAME = re.compile(r"《([^》]{2,60})》")


@dataclass(frozen=True)
class Citation:
    """一条引用。`article` 为归一化后的阿拉伯数字（无法解析时为 None）。"""

    law: str | None
    article: int | None
    article_raw: str
    item: int | None = None
    item_kind: str | None = None  # "项" | "款"

    @property
    def key(self) -> tuple[int | None, int | None, str | None]:
        return (self.article, self.item, self.item_kind)

    def __str__(self) -> str:
        parts = []
        if self.law:
            parts.append(f"《{self.law}》")
        parts.append(f"第{self.article}条" if self.article is not None
                     else f"第{self.article_raw}条")
        if self.item is not None:
            parts.append(f"第{self.item}{self.item_kind or '项'}")
        return "".join(parts)


def normalize_article_text(text: str) -> str:
    """把文本里的条号统一成阿拉伯数字写法（`第五百七十七条` → `第577条`）。

    用于**比较前的两侧归一化**：语料侧与答案侧都过一遍，再判命中。
    """
    def _repl(m: re.Match) -> str:
        n = cn2arabic(m.group(1))
        return f"第{n}条" if n is not None else m.group(0)

    out = RE_ARTICLE.sub(_repl, text or "")

    def _repl_k(m: re.Match) -> str:
        n = cn2arabic(m.group(1))
        return f"第{n}{m.group(2)}" if n is not None else m.group(0)

    out = RE_ITEM_KUAN.sub(_repl_k, out)

    def _repl_p(m: re.Match) -> str:
        n = cn2arabic(m.group(1))
        return f"（{n}）" if n is not None else m.group(0)

    return RE_PAREN_ITEM.sub(_repl_p, out)


def extract_citations(text: str, *, default_law: str | None = None) -> list[Citation]:
    """从答案文本里抽引用。

    策略：按句子切分，逐句找 `第X条`；若同句内出现 `《法名》` 则绑定该法名，
    否则用 `default_law`（上下文里最近一次出现的法名）兜底。

    项号只在**紧跟条号**（同一句、条号之后 12 字内）时绑定，避免把
    "（一）……（二）……" 这类并列枚举误挂到错误的条上。
    """
    out: list[Citation] = []
    current_law = default_law
    for sentence in re.split(r"[。；;\n]+", text or ""):
        laws = RE_LAW_NAME.findall(sentence)
        if laws:
            current_law = laws[-1]
        for m in RE_ARTICLE.finditer(sentence):
            art = cn2arabic(m.group(1))
            tail = sentence[m.end(): m.end() + 12]
            item = None
            kind = None
            mi = RE_ITEM_KUAN.search(tail)
            if mi:
                item = cn2arabic(mi.group(1))
                kind = mi.group(2)
            else:
                mp = RE_PAREN_ITEM.search(tail)
                if mp:
                    item = cn2arabic(mp.group(1))
                    kind = "项"
            out.append(Citation(
                law=current_law,
                article=art,
                article_raw=m.group(1),
                item=item,
                item_kind=kind,
            ))
    return out


def article_hit(answer_citations: list[Citation], context: str) -> tuple[int, int]:
    """返回 ``(命中数, 总数)``：答案里的条号有多少能在 ``context`` 中找到。

    ``context`` 会先做条号归一化，因此 `第577条` 与 `第五百七十七条` 视为同一个。
    """
    ctx = normalize_article_text(context or "")
    total = len(answer_citations)
    hit = 0
    for c in answer_citations:
        if c.article is None:
            continue
        if f"第{c.article}条" in ctx:
            hit += 1
    return hit, total
