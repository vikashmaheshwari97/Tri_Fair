"""Construct configured vLLM-backed Promptolution language models.

Torch is imported only when a model is actually created. Consequently,
configuration inspection and ``--help`` commands work on CPU-only Windows
machines, while real inference still requires Linux, CUDA, PyTorch and vLLM.
"""

from __future__ import annotations

import importlib
from typing import Any

from promptolution.llms import VLLM

from src.config.base_config import ModelConfig


def _load_torch():
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "PyTorch could not be imported. Tri-Fair model execution requires a working "
            "PyTorch/CUDA installation and is intended for the Linux GPU cluster. "
            "Metric tests and CLI --help do not require torch."
        ) from error


def get_available_gpu_memory_gb(device_index: int = 0) -> float:
    torch = _load_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to create the configured vLLM model")
    properties = torch.cuda.get_device_properties(device_index)
    return float(properties.total_memory) / (1024**3)


def create_llm(model_config: ModelConfig, seed: int) -> VLLM:
    """Create one vLLM wrapper without mutating the shared ModelConfig."""

    if not model_config.model.startswith("vllm-"):
        raise ValueError(
            f"Unsupported model type {model_config.model!r}; only vllm-* models are supported"
        )

    model_name = model_config.model.removeprefix("vllm-")
    available_memory_gb = get_available_gpu_memory_gb()
    optimal_batch_size = max(
        1,
        int(round(model_config.batch_size * min(1.0, available_memory_gb / 80.0))),
    )

    llm_kwargs: dict[str, Any] = dict(model_config.llm_kwargs)
    gpu_utilization = float(llm_kwargs.pop("gpu_memory_utilization", 0.90))
    tensor_parallel_size = int(llm_kwargs.pop("tensor_parallel_size", 1))
    dtype = str(llm_kwargs.pop("dtype", "auto"))
    trust_remote_code = bool(llm_kwargs.pop("trust_remote_code", False))

    return VLLM(
        model_name,
        batch_size=optimal_batch_size,
        max_model_len=model_config.max_model_len,
        model_storage_path=str(model_config.model_storage_path),
        seed=int(seed),
        gpu_memory_utilization=gpu_utilization,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        llm_kwargs=llm_kwargs,
    )
