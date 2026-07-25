# Tri-Fair v3 dataset-cache hotfix

This fixes the second Qwen failure in which the model loaded correctly from the
local snapshot, but `hf_hub_download` could not locate `heegyu/bbq`.

Cause: the previous model-cache helper set `HF_HUB_CACHE` to the Qwen-specific
model directory. That hid the normal Hugging Face dataset cache.

The replacement helper:
- keeps Qwen/GPT-OSS model loading on absolute local snapshot paths;
- resets Hugging Face Hub access to `~/.cache/huggingface/hub`;
- resets the `datasets` Arrow cache to `~/.cache/huggingface/datasets`;
- supports an override through `TF_HF_HUB_CACHE`;
- preserves offline execution after the required datasets are prefetched.
