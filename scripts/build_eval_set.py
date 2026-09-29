# -*- coding: utf-8 -*-
"""构建评测集 v1（D3）：六类题 + 反泄漏 manifest + 可复现。

六类（PLAN.md D3 / BLUEPRINT 第三章）
------------------------------------
  A 法条精确  60  从语料抽条文；判分=文件+条号双命中；同源=强（**仅回归基线**）
  B 口语化咨询 40  手写种子 + **原文校验**（源法存在且关键词命中，否则丢弃）；判分=期望来源召回；同源=中
  C 跨法条关联 20  手写种子（≥2 来源）+ 原文校验；判分=≥2 来源同时命中；同源=弱
  D 版本冲突  40  从多版本标题抽，3 难度梯度；判分=返回版本 status_code==3 / 时间旅行命中；同源=弱（**核心**）
  E 位阶/未生效 20  status_code=4 未生效 + 法律>行政法规>司法解释 位阶；同源=弱（**核心**）
  F 拒答      30  语料外 + status_code 缺失；判分=双侧（dev 标定 / test 报告）；同源=弱

反泄漏（D-08 附）
----------------
  - v1.jsonl 一旦存在**不就地覆盖**（需 --force 或换 --version）——旧项目同名覆盖三次导致结果不可复现。
  - manifest 记录：git commit + 脚本 sha256 + 内容 sha256 + seed + 各类 n + 同源等级 + 冻结时间。

用法
----
    python scripts/build_eval_set.py --dry-run     # 只统计各类可生成数量，不写盘
    python scripts/build_eval_set.py               # 生成 eval/sets/v1.jsonl + v1.manifest.json
    python scripts/build_eval_set.py --seed 42 --version v1
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import io
import json
import random
import re
import subprocess
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "npc"
DOCS = ROOT / "data" / "docs"
OUT_DIR = ROOT / "eval" / "sets"

CATS = ("法律", "行政法规", "司法解释")
DOMAIN_OF = {"法律": "statute", "行政法规": "statute", "司法解释": "interpretation"}
# 位阶：数字越小位阶越高
RANK = {"法律": 1, "行政法规": 2, "司法解释": 3}
STATUS_MEANING = {1: "已废止", 2: "已被修订", 3: "现行有效", 4: "尚未生效", None: "未标注", -1: "未知"}

CN_NUM = "一二三四五六七八九十百千零〇0-9"
RE_TIAO = re.compile(rf"\*\*(第[{CN_NUM}]+条)\*\*")


# ══════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════

def slugify(title: str, version: str) -> str:
    """必须与 ingest_npc.slugify 完全一致，否则定位不到 doc 文件。"""
    bad = '<>:"/\\|?*\n\r\t'
    t = "".join(c for c in title if c not in bad).strip()
    return f"{t[:70]}_{version}"


def parse_date(s):
    try:
        return datetime.date.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for cat in CATS:
        p = RAW / cat / "metadata.jsonl"
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["_cat"] = cat
                r["_domain"] = DOMAIN_OF[cat]
                r["_path"] = DOCS / r["_domain"] / f"{slugify(r['title'], r['bbbs'])}.md"
                r["_exists"] = r["_path"].is_file()
                rows.append(r)
    return rows


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def articles_of(path: Path) -> list[tuple[str, str]]:
    """从 doc 正文抽 [(条号, 该条正文前 120 字)]。"""
    if not path.is_file():
        return []
    txt = path.read_text(encoding="utf-8", errors="replace")
    # 去掉 front-matter
    if txt.startswith("---"):
        end = txt.find("\n---", 3)
        if end != -1:
            txt = txt[end + 4:]
    out: list[tuple[str, str]] = []
    marks = list(RE_TIAO.finditer(txt))
    for i, m in enumerate(marks):
        start = m.end()
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(txt)
        body = re.sub(r"\s+", " ", txt[start:stop]).strip()
        out.append((m.group(1), body[:120]))
    return out


# ══════════════════════════════════════════════════════════════
# 各类生成器
# ══════════════════════════════════════════════════════════════

def gen_A(rows, rng, n=60) -> list[dict]:
    """法条精确：现行有效（status_code=3）文档里抽一个实质条文。"""
    pool = [r for r in rows if r.get("status_code") == 3 and r["_exists"]]
    rng.shuffle(pool)
    items = []
    for r in pool:
        if len(items) >= n:
            break
        arts = articles_of(r["_path"])
        # 跳过"第一条…根据…制定本"这类目的条款，取有实义的条
        cand = [(t, b) for t, b in arts if not re.search(r"制定本|为了.*根据", b[:40])]
        if not cand:
            continue
        tiao, body = cand[rng.randrange(len(cand))]
        items.append({
            "category": "A", "difficulty": "exact_article", "same_source_level": "强",
            "question": f"《{r['title']}》{tiao}规定了什么内容？",
            "gt": {"expected_file": rel(r["_path"]), "expected_title": r["title"],
                   "article": tiao, "answer_prefix": body[:60]},
            "judge": {"type": "file_and_article_hit"},
            "provenance": "auto:corpus_article",
        })
    return items


def gen_D(rows, rng, n=40) -> list[dict]:
    """版本冲突：从多版本标题抽，3 难度梯度（简单=现行版 / 中等=列全部版本 / 困难=时间旅行）。"""
    by_title = collections.defaultdict(list)
    for r in rows:
        if r["_exists"]:
            by_title[(r["_cat"], r["title"])].append(r)
    multi = {k: v for k, v in by_title.items() if len(v) > 1}
    keys = sorted(multi, key=lambda k: k[1])
    rng.shuffle(keys)

    easy, mid, hard = [], [], []
    n_easy = n // 2            # 20
    n_mid = n - n_easy - n // 4  # 10
    for k in keys:
        vs = multi[k]
        cat, title = k
        cur = [r for r in vs if r.get("status_code") == 3]
        if not cur:
            continue
        cur = cur[0]
        # 简单：现行有效版
        if len(easy) < n_easy:
            arts = articles_of(cur["_path"])
            tiao = arts[0][0] if arts else "第一条"
            easy.append({
                "category": "D", "difficulty": "version_current", "same_source_level": "弱",
                "question": f"《{title}》{tiao}现行有效的规定是什么？",
                "gt": {"expected_file": rel(cur["_path"]), "expected_bbbs": cur["bbbs"],
                       "expected_status_code": 3, "n_versions": len(vs)},
                "judge": {"type": "top1_status_code_eq", "value": 3},
                "provenance": "auto:multi_version",
            })
        # 中等：列出全部版本
        elif len(mid) < n_mid:
            mid.append({
                "category": "D", "difficulty": "version_enumerate", "same_source_level": "弱",
                "question": f"《{title}》一共有几个版本？各自的效力状态如何？",
                "gt": {"n_versions": len(vs),
                       "versions": [{"bbbs": v["bbbs"], "status_code": v.get("status_code"),
                                     "status_label": STATUS_MEANING.get(v.get("status_code")),
                                     "effective_date": v.get("effective_date")} for v in
                                    sorted(vs, key=lambda x: x.get("effective_date") or "")]},
                "judge": {"type": "version_set_recall"},
                "provenance": "auto:multi_version",
            })
    # 困难：时间旅行（在旧版生效、新版尚未生效之间的年份）
    for k in keys:
        if len(hard) >= n - n_easy - n_mid:
            break
        vs = sorted(multi[k], key=lambda r: parse_date(r.get("effective_date")) or datetime.date(1900, 1, 1))
        dated = [r for r in vs if parse_date(r.get("effective_date"))]
        if len(dated) < 2:
            continue
        old, new = dated[0], dated[-1]
        d_old, d_new = parse_date(old["effective_date"]), parse_date(new["effective_date"])
        if d_new.year - d_old.year < 2:
            continue
        probe_year = (d_old.year + d_new.year) // 2
        # 该年生效的版本 = effective_date <= probe_year-12-31 里最新的一个
        asof = datetime.date(probe_year, 12, 31)
        effective_then = [r for r in dated if parse_date(r["effective_date"]) <= asof]
        if not effective_then:
            continue
        gt_ver = max(effective_then, key=lambda r: parse_date(r["effective_date"]))
        hard.append({
            "category": "D", "difficulty": "version_time_travel", "same_source_level": "弱",
            "question": f"在 {probe_year} 年时，《{k[1]}》适用的是哪一版？当时是怎么规定的？",
            "gt": {"probe_year": probe_year, "expected_bbbs": gt_ver["bbbs"],
                   "expected_effective_date": gt_ver.get("effective_date"),
                   "expected_file": rel(gt_ver["_path"])},
            "judge": {"type": "version_effective_at_year", "year": probe_year},
            "provenance": "auto:multi_version_time_travel",
        })
    return easy + mid + hard


def gen_E(rows, rng, n=20) -> list[dict]:
    """位阶/未生效：status_code=4 未生效 + 法律>行政法规>司法解释 位阶冲突。"""
    items = []
    # ① 未生效
    pending = [r for r in rows if r.get("status_code") == 4 and r["_exists"]]
    rng.shuffle(pending)
    n_pending = min(len(pending), n // 2 + 2)
    by_title = collections.defaultdict(list)
    for r in rows:
        by_title[(r["_cat"], r["title"])].append(r)
    for r in pending[:n_pending]:
        cur = [x for x in by_title[(r["_cat"], r["title"])] if x.get("status_code") == 3 and x["_exists"]]
        items.append({
            "category": "E", "difficulty": "not_yet_effective", "same_source_level": "弱",
            "question": f"《{r['title']}》现在生效了吗？什么时候施行？",
            "gt": {"status_code": 4, "status_label": "尚未生效",
                   "effective_date": r.get("effective_date"),
                   "current_version_file": rel(cur[0]["_path"]) if cur else None,
                   "expected_file": rel(r["_path"])},
            "judge": {"type": "detect_not_effective"},
            "provenance": "auto:status_code_4",
        })
    # ② 位阶冲突：《X法》与《X法实施条例/实施细则》——后者标题以前者开头，保证**同主题**
    titles = collections.defaultdict(list)
    for r in rows:
        if r["_exists"]:
            titles[r["title"]].append(r)
    pairs = []
    seen = set()
    for t in titles:
        if not (t.endswith("实施条例") or t.endswith("实施细则")):
            continue
        r_impl = titles[t][0]
        # 父法候选：标题是 t 的前缀（取最长、最具体的那个）
        cands = sorted((bt for bt in titles if bt != t and t.startswith(bt)), key=len, reverse=True)
        for bt in cands:
            r_base = titles[bt][0]
            if RANK.get(r_base["_cat"], 9) < RANK.get(r_impl["_cat"], 9):
                key = (r_base["bbbs"], r_impl["bbbs"])
                if key not in seen:
                    seen.add(key)
                    pairs.append((r_base, r_impl))
                break
    rng.shuffle(pairs)
    for r_base, r_impl in pairs:
        if len(items) >= n:
            break
        items.append({
            "category": "E", "difficulty": "hierarchy_conflict", "same_source_level": "弱",
            "question": f"《{r_base['title']}》和《{r_impl['title']}》规定不一致时，应以哪个为准？为什么？",
            "gt": {"higher_rank_file": rel(r_base["_path"]), "higher_rank_title": r_base["title"],
                   "higher_rank_nature": r_base["_cat"], "lower_rank_nature": r_impl["_cat"],
                   "rank_order": ["法律", "行政法规", "司法解释"]},
            "judge": {"type": "hierarchy_higher_wins"},
            "provenance": "auto:hierarchy_pair",
        })
    return items[:n]


# F 类：语料外（应拒答）—— 明确不在法律语料内的问题
F_OUT_OF_CORPUS = [
    "美国宪法第一修正案的具体内容是什么？",
    "法国劳动法关于解雇的规定是什么？",
    "如何用 Python 实现快速排序？",
    "今天上海的天气怎么样？",
    "iPhone 17 的售价是多少？",
    "《联合国海洋法公约》第 89 条讲了什么？",
    "日本民法典关于收养的规定？",
    "帮我写一首关于秋天的诗",
    "特斯拉 2025 年的财报营收是多少？",
    "英国脱欧公投是哪一年？",
    "德国《基本法》第一条的内容？",
    "红楼梦的作者是谁？",
    "如何评价某位明星的演技？",
    "新加坡公司法对董事的要求？",
    "世界杯历届冠军有哪些？",
]


def gen_F(rows, rng, n=30) -> list[dict]:
    """拒答：语料外（15）+ status_code 缺失（15），dev/test 各半。"""
    items = []
    ooc = list(F_OUT_OF_CORPUS)
    rng.shuffle(ooc)
    for q in ooc[: n // 2]:
        items.append({
            "category": "F", "difficulty": "out_of_corpus", "same_source_level": "弱",
            "question": q,
            "gt": {"should_refuse": True, "reason": "out_of_corpus"},
            "judge": {"type": "should_refuse"},
            "provenance": "manual:out_of_corpus",
        })
    unlabeled = [r for r in rows if r.get("status_code") is None and r["_exists"]]
    rng.shuffle(unlabeled)
    for r in unlabeled[: n - len(items)]:
        items.append({
            "category": "F", "difficulty": "status_unlabeled", "same_source_level": "弱",
            "question": f"《{r['title']}》现在是否现行有效？",
            "gt": {"should_disclaim": True, "reason": "status_unlabeled",
                   "status_code": None, "expected_file": rel(r["_path"])},
            "judge": {"type": "should_disclaim_unknown_effectiveness"},
            "provenance": "auto:status_code_none",
        })
    # dev/test 拆分（D-08 附：dev 标定 / test 报告）
    for i, it in enumerate(items):
        it["split"] = "dev" if i % 2 == 0 else "test"
    return items


# B 类：口语化咨询（手写 + 原文校验）——(问题, 期望法名, 校验关键词)
B_SEEDS = [
    ("公司一直拖着不发工资，我能怎么办？", "中华人民共和国劳动合同法", "劳动报酬|工资|拖欠"),
    ("试用期最长能约定多久啊？", "中华人民共和国劳动合同法", "试用期"),
    ("公司没跟我签书面劳动合同，要紧吗？", "中华人民共和国劳动合同法", "书面劳动合同|订立书面"),
    ("我被公司辞退了，有没有经济补偿？", "中华人民共和国劳动合同法", "经济补偿"),
    ("加班费应该怎么算？", "中华人民共和国劳动法", "延长工作时间|加班|工资报酬"),
    ("公司能不给我交社保吗？", "中华人民共和国劳动法", "社会保险"),
    ("签了竞业限制协议，离职后公司要给补偿吗？", "中华人民共和国劳动合同法", "竞业限制"),
    ("合同里没写违约金，对方违约了能要赔偿吗？", "中华人民共和国民法典", "违约金|损失赔偿"),
    ("租房合同没到期房东赶我走，怎么办？", "中华人民共和国民法典", "租赁|承租人"),
    ("买的房子交了定金，对方反悔了定金能退吗？", "中华人民共和国民法典", "定金"),
    ("借钱给别人没写借条，能要回来吗？", "中华人民共和国民法典", "借款"),
    ("网购的东西有质量问题，能要求退货吗？", "中华人民共和国消费者权益保护法", "退货|质量|经营者"),
    ("遇到霸王条款，商家说的免责有效吗？", "中华人民共和国消费者权益保护法", "格式条款|不公平|免责"),
    ("股东想退出公司，股份怎么处理？", "中华人民共和国公司法", "股权转让|股东"),
    ("夫妻离婚，孩子抚养权怎么判？", "中华人民共和国民法典", "抚养|离婚"),
    ("遗产没有遗嘱，怎么分？", "中华人民共和国民法典", "继承|法定继承"),
    ("工伤了公司不认，怎么申请认定？", "工伤保险条例", "工伤认定"),
    ("不动产过户要办什么登记？", "不动产登记暂行条例", "登记"),
    ("用人单位能随便调我的岗位吗？", "中华人民共和国劳动合同法", "变更劳动合同|工作岗位"),
    ("产假有多少天？", "女职工劳动保护特别规定", "产假"),
    ("公司以经营困难为由裁员，合法吗？", "中华人民共和国劳动合同法", "裁减人员|经济性裁员"),
    ("未成年工有什么特殊保护？", "中华人民共和国劳动法", "未成年工"),
    ("欠条和借条有什么区别，诉讼时效多久？", "中华人民共和国民法典", "诉讼时效"),
    ("物业不作为，业主能换物业吗？", "中华人民共和国民法典", "物业服务|业主"),
    ("民间借贷利息最高能约定多少？", "中华人民共和国民法典", "借款|利息"),
    ("合同签了但对方盖的是假章，合同有效吗？", "中华人民共和国民法典", "代理|盖章|效力"),
    ("公司注销了，欠我的工资找谁要？", "中华人民共和国公司法", "清算|注销"),
    ("买了预售房烂尾了怎么办？", "中华人民共和国民法典", "商品房|买卖"),
    ("劳动合同到期公司不续签，有补偿吗？", "中华人民共和国劳动合同法", "劳动合同期满|经济补偿"),
    ("上班路上出车祸算工伤吗？", "工伤保险条例", "上下班途中|工伤"),
    ("小区电梯广告收益归谁？", "中华人民共和国民法典", "共有|收益"),
    ("对方违约，我能同时要求违约金和定金吗？", "中华人民共和国民法典", "违约金|定金"),
    ("公司注销前债务怎么清偿？", "中华人民共和国公司法", "清算|债务"),
    ("非全日制用工可以随时终止吗？", "中华人民共和国劳动合同法", "非全日制"),
    ("赠与人能反悔撤销赠与吗？", "中华人民共和国民法典", "赠与|撤销"),
    ("劳务派遣工和正式工待遇一样吗？", "中华人民共和国劳动合同法", "劳务派遣"),
    ("高空抛物砸到人，谁负责？", "中华人民共和国民法典", "高空|抛掷|建筑物"),
    ("网络服务提供者对侵权内容要担责吗？", "中华人民共和国民法典", "网络服务提供者|侵权"),
    ("债权人能把债权转让给别人吗？", "中华人民共和国民法典", "债权转让"),
    ("公司对外担保需要什么程序？", "中华人民共和国公司法", "担保"),
]

# C 类：跨法条关联（≥2 来源）——(问题, [期望法名...], 校验关键词)
C_SEEDS = [
    ("关于试用期，法律是怎么规定的？", ["中华人民共和国劳动合同法", "中华人民共和国劳动法"], "试用期"),
    ("违约金能随便约定吗？有哪些限制？", ["中华人民共和国民法典", "中华人民共和国劳动合同法"], "违约金"),
    ("经济补偿和赔偿金有什么区别？", ["中华人民共和国劳动合同法", "中华人民共和国劳动合同法实施条例"], "经济补偿|赔偿"),
    ("工资支付有哪些法律要求？", ["中华人民共和国劳动法", "中华人民共和国劳动合同法"], "工资|劳动报酬"),
    ("不动产登记的效力是怎么规定的？", ["中华人民共和国民法典", "不动产登记暂行条例"], "登记"),
    ("关于劳动合同的解除，都有哪些规定？", ["中华人民共和国劳动合同法", "中华人民共和国劳动法"], "解除劳动合同"),
    ("担保的法律规定散见于哪些法？", ["中华人民共和国民法典", "中华人民共和国公司法"], "担保"),
    ("工伤认定和赔偿依据哪些规定？", ["工伤保险条例", "中华人民共和国劳动法"], "工伤"),
    ("消费者权益受哪些法律保护？", ["中华人民共和国消费者权益保护法", "中华人民共和国民法典"], "消费者|经营者"),
    ("夫妻财产和债务，法律是怎么规定的？", ["中华人民共和国民法典", "中华人民共和国婚姻法"], "夫妻"),
    ("继承相关的规定有哪些？", ["中华人民共和国民法典", "中华人民共和国婚姻法"], "继承"),
    ("无固定期限劳动合同在什么情况下要签？", ["中华人民共和国劳动合同法", "中华人民共和国劳动合同法实施条例"], "无固定期限"),
    ("社会保险涉及哪些法规？", ["中华人民共和国劳动法", "中华人民共和国社会保险法"], "社会保险"),
    ("合同无效的法定情形有哪些？", ["中华人民共和国民法典", "中华人民共和国劳动合同法"], "无效"),
    ("关于加班和工时，法律怎么规定？", ["中华人民共和国劳动法", "中华人民共和国劳动合同法"], "工作时间|加班"),
    ("哪些格式条款（霸王条款）是无效的？", ["中华人民共和国民法典", "中华人民共和国消费者权益保护法"], "格式条款"),
    ("用人单位的规章制度有什么法律要求？", ["中华人民共和国劳动合同法", "中华人民共和国劳动法"], "规章制度"),
    ("未成年人保护涉及哪些法律规定？", ["中华人民共和国民法典", "中华人民共和国劳动法"], "未成年"),
    ("借款合同的主要法律规定？", ["中华人民共和国民法典", "中华人民共和国公司法"], "借款"),
    ("法人的民事权利能力和责任怎么规定？", ["中华人民共和国民法典", "中华人民共和国公司法"], "法人"),
]


def _resolve_current(title: str, by_title: dict):
    """返回该法名现行有效（status_code=3）且存在的 doc；无则退回任一存在版本。"""
    vs = by_title.get(title, [])
    vs = [r for r in vs if r["_exists"]]
    if not vs:
        return None
    cur = [r for r in vs if r.get("status_code") == 3]
    return (cur or vs)[0]


def gen_B(by_title, rng, n=40) -> tuple[list[dict], list[str]]:
    """口语化咨询：手写种子 + 原文校验（源法存在 + 关键词命中正文）。"""
    items, dropped = [], []
    for q, law, kw in B_SEEDS:
        r = _resolve_current(law, by_title)
        if r is None:
            dropped.append(f"{law}（源法不在库/文件缺失）")
            continue
        txt = r["_path"].read_text(encoding="utf-8", errors="replace")
        if not re.search(kw, txt):
            dropped.append(f"{law}（关键词 /{kw}/ 未命中，需人工核对）")
            continue
        items.append({
            "category": "B", "difficulty": "colloquial", "same_source_level": "中",
            "question": q,
            "gt": {"expected_file": rel(r["_path"]), "expected_title": law, "verify_keyword": kw},
            "judge": {"type": "expected_source_recall"},
            "provenance": "manual_seed:verified",
        })
        if len(items) >= n:
            break
    return items, dropped


def gen_C(by_title, rng, n=20) -> tuple[list[dict], list[str]]:
    """跨法条关联：手写种子（≥2 来源）+ 原文校验。"""
    items, dropped = [], []
    for q, laws, kw in C_SEEDS:
        resolved, missing = [], []
        for law in laws:
            r = _resolve_current(law, by_title)
            if r is None:
                missing.append(law)
                continue
            txt = r["_path"].read_text(encoding="utf-8", errors="replace")
            if not re.search(kw, txt):
                missing.append(f"{law}(kw未中)")
                continue
            resolved.append(r)
        if len(resolved) < 2:
            dropped.append(f"{q[:16]}…（有效来源<2：{missing}）")
            continue
        items.append({
            "category": "C", "difficulty": "cross_article", "same_source_level": "弱",
            "question": q,
            "gt": {"expected_files": [rel(r["_path"]) for r in resolved],
                   "expected_titles": [r["title"] for r in resolved],
                   "min_sources": 2, "verify_keyword": kw},
            "judge": {"type": "multi_source_recall", "min_sources": 2},
            "provenance": "manual_seed:verified",
        })
        if len(items) >= n:
            break
    return items, dropped


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="构建评测集 v1（六类 + 反泄漏 manifest）")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="允许覆盖已存在的评测集（默认禁止）")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_rows()
    if not rows:
        print("❌ 未读到语料元数据（data/raw/npc/*/metadata.jsonl）", file=sys.stderr)
        return 1
    by_title = collections.defaultdict(list)
    for r in rows:
        by_title[r["title"]].append(r)
    print(f"载入元数据 {len(rows)} 条；其中磁盘存在 {sum(1 for r in rows if r['_exists'])} 条；唯一标题 {len(by_title)}")

    A = gen_A(rows, rng, 60)
    D = gen_D(rows, rng, 40)
    E = gen_E(rows, rng, 20)
    F = gen_F(rows, rng, 30)
    B, b_drop = gen_B(by_title, rng, 40)
    C, c_drop = gen_C(by_title, rng, 20)

    allitems = A + B + C + D + E + F
    for i, it in enumerate(allitems):
        it["id"] = f"{it['category']}-{i:04d}"

    counts = collections.Counter(it["category"] for it in allitems)
    print("\n各类生成数量：")
    for c in "ABCDEF":
        print(f"  {c}: {counts.get(c, 0)}")
    if b_drop:
        print(f"\nB 类丢弃 {len(b_drop)}（原文校验未过）：{b_drop[:6]}")
    if c_drop:
        print(f"C 类丢弃 {len(c_drop)}（有效来源<2）：{c_drop[:6]}")

    if args.dry_run:
        print("\n(dry-run) 未写盘")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / f"{args.version}.jsonl"
    manifest_path = OUT_DIR / f"{args.version}.manifest.json"
    if jsonl_path.exists() and not args.force:
        print(f"\n❌ {rel(jsonl_path)} 已存在。评测集冻结后不得就地覆盖（D-08 附）。"
              f"\n   要重生成：换 --version（如 v2），或加 --force（会丢失可复现性，慎用）。", file=sys.stderr)
        return 2

    body = "\n".join(json.dumps(it, ensure_ascii=False) for it in allitems) + "\n"
    jsonl_path.write_text(body, encoding="utf-8")
    content_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    script_sha = hashlib.sha256((ROOT / "scripts" / "build_eval_set.py").read_bytes()).hexdigest()

    manifest = {
        "version": args.version,
        "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "git_commit": _git_commit(),
        "script_sha256": script_sha,
        "content_sha256": content_sha,
        "total": len(allitems),
        "per_category": {c: counts.get(c, 0) for c in "ABCDEF"},
        "same_source_level": {
            "A": "强（仅回归基线，不当能力证据）", "B": "中", "C": "弱",
            "D": "弱（核心）", "E": "弱（核心）", "F": "弱",
        },
        "splits": {
            "F_dev": sum(1 for it in F if it.get("split") == "dev"),
            "F_test": sum(1 for it in F if it.get("split") == "test"),
        },
        "d_difficulty": dict(collections.Counter(it["difficulty"] for it in D)),
        "b_dropped": b_drop,
        "c_dropped": c_drop,
        "judge_types": sorted({it["judge"]["type"] for it in allitems}),
        "note": "A 类同源强，只作回归基线；B/C 为手写种子经原文校验；D/E 为核心可自动判分题。"
                "评测集冻结后不得就地覆盖（见 why/04）。",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 写出 {rel(jsonl_path)}（{len(allitems)} 题）+ {rel(manifest_path)}")
    print(f"   content_sha256={content_sha[:16]}… git={manifest['git_commit'][:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
