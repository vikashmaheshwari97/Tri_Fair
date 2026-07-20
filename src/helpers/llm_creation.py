"""Construct configured vLLM-backed Promptolution language models.

Torch is imported only when a model is actually created. Consequently,
configuration inspection and ``--help`` commands work on CPU-only Windows
machines, while real inference still requires Linux, CUDA, PyTorch and vLLM.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from promptolution.llms import VLLM

from src.config.base_config import ModelConfig

GPT_OSS_ALIAS = "gpt-oss-120b"
GPT_OSS_REQUIRED_SHARDS = 15
GPT_OSS_REASONING_EFFORTS = frozenset({"low", "medium", "high"})


class GPTOSSVLLM(VLLM):
    """Promptolution vLLM wrapper with GPT-OSS reasoning-effort rendering."""

    def __init__(
        self,
        *args: Any,
        reasoning_effort: str = "low",
        **kwargs: Any,
    ) -> None:
        normalized = str(reasoning_effort).strip().casefold()
        if normalized not in GPT_OSS_REASONING_EFFORTS:
            raise ValueError(
                "GPT-OSS reasoning_effort must be one of "
                f"{sorted(GPT_OSS_REASONING_EFFORTS)}, got {reasoning_effort!r}"
            )
        self.reasoning_effort = normalized
        super().__init__(*args, **kwargs)

    def _get_response(
        self,
        prompts: list[str],
        system_prompts: list[str],
    ) -> list[str]:
        """Render GPT-OSS prompts with an explicit reasoning effort."""

        rendered_prompts = [
            str(
                self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    reasoning_effort=self.reasoning_effort,
                )
            )
            for prompt, system_prompt in zip(prompts, system_prompts)
        ]

        responses: list[str] = []
        for index in range(0, len(rendered_prompts), self.batch_size):
            batch = rendered_prompts[index : index + self.batch_size]
            outputs = self.llm.generate(batch, self.sampling_params)
            responses.extend(output.outputs[0].text for output in outputs)
        return responses


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


def _resolve_gpt_oss_snapshot() -> Path:
    raw = os.environ.get("GPT_OSS_LOCAL_SNAPSHOT", "").strip()
    if not raw:
        raise RuntimeError(
            "GPT_OSS_LOCAL_SNAPSHOT must point to the verified local GPT-OSS-120B "
            "snapshot directory"
        )

    snapshot = Path(raw).expanduser().resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(
            f"GPT_OSS_LOCAL_SNAPSHOT is not a directory: {snapshot}"
        )

    required_files = (
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    )
    missing = [name for name in required_files if not (snapshot / name).is_file()]
    if missing:
        raise RuntimeError(
            f"GPT-OSS local snapshot is missing required files: {missing}"
        )

    shards = sorted(snapshot.glob("model-*.safetensors"))
    if len(shards) != GPT_OSS_REQUIRED_SHARDS:
        raise RuntimeError(
            "GPT-OSS local snapshot must contain exactly "
            f"{GPT_OSS_REQUIRED_SHARDS} root safetensor shards; found {len(shards)}"
        )
    return snapshot


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

    wrapper_type: type[VLLM] = VLLM
    wrapper_kwargs: dict[str, Any] = {}
    if model_config.alias == GPT_OSS_ALIAS:
        model_name = str(_resolve_gpt_oss_snapshot())
        # A local snapshot has no Hugging Face revision to resolve.
        llm_kwargs.pop("revision", None)
        wrapper_type = GPTOSSVLLM
        wrapper_kwargs["reasoning_effort"] = "low"

    return wrapper_type(
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
        **wrapper_kwargs,
    )
