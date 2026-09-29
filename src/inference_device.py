"""本地 PyTorch 推理设备：auto 时优先 CUDA，否则 CPU（无 CUDA 或仅 CPU 版 torch 时自动回退）。"""
from __future__ import annotations

import logging
from typing import Any

from src.config_LEGACY_REFERENCE import Settings

logger = logging.getLogger(__name__)

_logged: set[str] = set()


def torch_cuda_built() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends.cuda, "is_built", lambda: False)())
    except Exception:  # noqa: BLE001
        return False


def cuda_device_info() -> dict[str, Any]:
    """供 /health/config 与排障；不触发模型加载。"""
    out: dict[str, Any] = {
        "torch_imported": False,
        "torch_version": None,
        "torch_cuda_built": False,
        "cuda_available": False,
        "cuda_version_built": None,
        "device_name": None,
        "device_capability": None,
    }
    try:
        import torch
    except ImportError:
        return out
    out["torch_imported"] = True
    out["torch_version"] = torch.__version__
    out["torch_cuda_built"] = torch_cuda_built()
    out["cuda_version_built"] = getattr(torch.version, "cuda", None)
    if not torch.cuda.is_available():
        return out
    out["cuda_available"] = True
    try:
        out["device_name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        out["device_capability"] = f"{cap[0]}.{cap[1]}"
    except Exception:  # noqa: BLE001
        pass
    return out


def _warn_if_gpu_present_but_cpu_torch() -> None:
    """RTX 50 等常见场景：已装 CPU 版 torch，导致永远走 CPU。"""
    info = cuda_device_info()
    if info.get("cuda_available"):
        return
    if not info.get("torch_imported"):
        return
    name = (info.get("device_name") or "").lower()
    if info.get("torch_cuda_built") is False or info.get("cuda_version_built") is None:
        logger.warning(
            "当前 PyTorch 为 CPU 构建（%s），无法使用外接 NVIDIA GPU。"
            "RTX 50 系需 cu128 wheel：在 conda 环境 rags 中运行 scripts/setup_pytorch_cuda.ps1",
            info.get("torch_version"),
        )
        return
    cap = info.get("device_capability") or ""
    if cap.startswith("12.") or "5070" in name or "5080" in name or "5090" in name:
        logger.warning(
            "检测到 GPU（%s, sm_%s）但 torch.cuda.is_available()=False；"
            "若为新显卡请确认已安装 cu128 版 PyTorch（见 scripts/setup_pytorch_cuda.ps1）",
            info.get("device_name") or "NVIDIA",
            cap.replace(".", ""),
        )


def resolve_inference_device(settings: Settings) -> str:
    """返回 ``cuda`` 或 ``cpu``（不写死；无 CUDA 时一律 cpu）。"""
    raw = (getattr(settings, "inference_device", None) or "auto").strip().lower()
    if raw not in ("auto", "cuda", "cpu"):
        logger.warning("未知 inference_device=%r，按 auto 处理", raw)
        raw = "auto"

    try:
        import torch
    except ImportError:
        if raw == "cuda":
            logger.warning("未安装 torch，无法使用 CUDA，使用 CPU")
        return "cpu"

    if raw == "auto":
        if torch.cuda.is_available():
            return "cuda"
        _warn_if_gpu_present_but_cpu_torch()
        return "cpu"
    if raw == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        _warn_if_gpu_present_but_cpu_torch()
        logger.warning("已配置 inference_device=cuda 但当前 CUDA 不可用，回退 CPU")
        return "cpu"
    return "cpu"


def log_device_context(context: str, device: str) -> None:
    """每个进程对每个 context 只打一条，避免刷屏。"""
    key = f"{context}:{device}"
    if key in _logged:
        return
    _logged.add(key)
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("%s: device=cuda (%s)", context, name)
        else:
            logger.info("%s: device=%s", context, device)
    except Exception:  # noqa: BLE001
        logger.info("%s: device=%s", context, device)
