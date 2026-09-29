# -*- coding: utf-8 -*-
"""全语料切块有界验证：7 批 × 500 篇，内置自看门狗（Commit > 3.5GB 自毁）。"""
import ctypes
import sys
import io
import os
import threading
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

LIMIT_MB = 3500

class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("pf", ctypes.c_ulong), ("pk", ctypes.c_size_t),
                ("ws", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t),
                ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("page", ctypes.c_size_t),
                ("pkp", ctypes.c_size_t)]

def commit_mb():
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.page / 2**20

def watchdog():
    while True:
        time.sleep(0.5)
        try:
            if commit_mb() > LIMIT_MB:
                print(f"SELF_WATCHDOG_KILL: Commit {commit_mb():.0f}MB 超 {LIMIT_MB}MB", flush=True)
                os._exit(42)
        except Exception:
            pass

threading.Thread(target=watchdog, daemon=True).start()

from src.config_LEGACY_REFERENCE import get_settings  # noqa: E402
from src.chunking import load_documents, build_nodes  # noqa: E402

s = get_settings().model_copy(update={"docs_dir": "data/docs/legal"})
docs = load_documents(Path("data/docs/legal"))
print(f"载入 {len(docs)} 篇, 原文总字数 {sum(len(d.text) for d in docs)}", flush=True)

BATCH = 500
total_nodes = 0
total_chars = 0
n_batches = (len(docs) + BATCH - 1) // BATCH
for i in range(n_batches):
    batch = docs[i * BATCH:(i + 1) * BATCH]
    nodes = build_nodes(batch, s)
    chars = sum(len(n.get_content()) for n in nodes)
    total_nodes += len(nodes)
    total_chars += chars
    print(f"批 {i+1}/{n_batches}: docs={len(batch)} nodes={len(nodes)} node_chars={chars} Commit={commit_mb():.0f}MB", flush=True)

src_chars = sum(len(d.text) for d in docs)
ratio = total_chars / src_chars if src_chars else 0
print(f"汇总: 总节点 {total_nodes} | 节点总字数 {total_chars} | 原文总字数 {src_chars} | 复制率 {ratio:.2f}（期望 1.0-1.3）", flush=True)
print("VERDICT:", "BOUNDED_OK" if ratio < 1.6 else "SUSPECT_DUPLICATION", flush=True)
