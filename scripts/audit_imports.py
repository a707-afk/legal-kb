# -*- coding: utf-8 -*-
"""审计 import 完整性：确认命名空间已统一到 src.*，且所有 src.* 引用都能解析到真实模块。

迁移前它统计「被引用但未迁移的 app.* 模块」；迁移完成后（app.* → src.*）改为：
  1) 报告任何残留的 app.* 引用（应为 0）；
  2) 报告指向不存在模块的 src.* 断链（应为 0）；
  3) 报告对已删除模块（一代残留 / 第五章「明确不做」）的引用（应为 0）。

用法：
    python scripts/audit_imports.py
退出码 0 = 干净；1 = 仍有断链/残留。
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 已明确删除、不应再被引用的模块前缀（一代客服残留 / OPA / Prometheus / Redis）
FORBIDDEN_PREFIX = (
    "src.domain_router", "src.retrieval_intent_boost",
    "src.router_calibration", "src.router_profiles",
    "src.behavior_guard", "src.agent.tools",
    "src.metrics", "src.policy", "src.redis_client",
)

IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([\w.]+)", re.M)


def _have_modules() -> set[str]:
    mods: set[str] = set()
    for p in Path("src").rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        m = str(p.with_suffix("")).replace("\\", ".").replace("/", ".")
        if m.endswith(".__init__"):
            m = m[: -len(".__init__")]
        mods.add(m)
    return mods


def main() -> int:
    have = _have_modules()
    files = [
        p for p in list(Path("src").rglob("*.py")) + list(Path("tests").rglob("*.py"))
        if "__pycache__" not in str(p)
    ]

    legacy: dict[str, set[str]] = {}    # 残留 app.* 引用
    broken: dict[str, set[str]] = {}    # src.* 指向不存在的模块
    forbidden: dict[str, set[str]] = {} # 引用了已删除模块

    for p in files:
        txt = p.read_text(encoding="utf-8", errors="replace")
        for m in IMPORT_RE.finditer(txt):
            mod = m.group(1)
            if mod == "app" or mod.startswith("app."):
                legacy.setdefault(mod, set()).add(str(p))
            elif mod.startswith("src."):
                if any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN_PREFIX):
                    forbidden.setdefault(mod, set()).add(str(p))
                elif mod not in have and not any(h.startswith(mod + ".") for h in have):
                    broken.setdefault(mod, set()).add(str(p))

    print(f"扫描 {len(files)} 个文件（src 模块 {len(have)} 个）")
    print(f"  残留 app.* 引用      : {len(legacy)} 个模块")
    for mod in sorted(legacy):
        print(f"    {mod} ← {sorted(legacy[mod])}")
    print(f"  src.* 断链           : {len(broken)} 个模块")
    for mod in sorted(broken):
        print(f"    {mod} ← {sorted(broken[mod])[:4]}")
    print(f"  引用已删除模块       : {len(forbidden)} 个模块")
    for mod in sorted(forbidden):
        print(f"    {mod} ← {sorted(forbidden[mod])[:4]}")

    if not (legacy or broken or forbidden):
        print("\n✅ 命名空间已统一到 src.*，无断链、无残留、无对已删模块的引用。")
        return 0
    print("\n❌ 仍有断链/残留，见上。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
