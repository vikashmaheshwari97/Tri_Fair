"""Load the configured Qwen model through Promptolution and vLLM."""

from __future__ import annotations

from multiprocessing import freeze_support
from pathlib import Path


def main() -> None:
    # Keep vLLM-related imports inside the guarded entry point. This allows
    # multiprocessing workers using "spawn" to import this module safely.
    from src.config.model_configs import ALL_MODELS
    from src.helpers.llm_creation import create_llm

    config = ALL_MODELS["qwen-3-30b"]

    print("Model:", config.model, flush=True)
    print("Revision:", config.llm_kwargs.get("revision"), flush=True)
    print(
        "Storage:",
        Path(config.model_storage_path).expanduser().resolve(),
        flush=True,
    )

    model = create_llm(config, seed=42)

    print("Wrapper:", type(model).__name__, flush=True)
    print("Qwen model initialization: PASSED", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()