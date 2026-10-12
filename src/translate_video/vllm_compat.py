"""Qwen tokenizer/config integration for the pinned vLLM 0.14 runtime."""

from vllm.tokenizers.hf import CachedHfTokenizer
from vllm.tokenizers.registry import TokenizerRegistry


class QwenTokenizer(CachedHfTokenizer):
    @classmethod
    def from_pretrained(cls, path_or_repo_id, *args, **kwargs):
        # Match the tokenizer correction used by Qwen's official ASR wrapper.
        kwargs["fix_mistral_regex"] = True
        return super().from_pretrained(path_or_repo_id, *args, **kwargs)


def register_qwen_tokenizer():
    mode = "translate_video_qwen"
    if mode not in TokenizerRegistry.tokenizers:
        TokenizerRegistry.register(mode, __name__, "QwenTokenizer")
    return mode


def qwen_inference_config(config):
    import torch

    # vLLM 0.14 otherwise asks the Hub for dtype metadata even for a local
    # snapshot without a dtype field. Use the same BF16 inference dtype that
    # the previous automatic configuration selected, without that Hub lookup.
    config.dtype = torch.bfloat16
    return config
