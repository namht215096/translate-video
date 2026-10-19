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
    try:
        # vLLM >= 0.2x resolves a renderer by the same mode name.
        from vllm.renderers.registry import RENDERER_REGISTRY

        if mode not in RENDERER_REGISTRY.renderers:
            RENDERER_REGISTRY.register(mode, "vllm.renderers.hf", "HfRenderer")
    except ImportError:
        pass
    return mode


def qwen_inference_config(config):
    import torch

    # vLLM 0.14 otherwise asks the Hub for dtype metadata even for a local
    # snapshot without a dtype field. Use the same BF16 inference dtype that
    # the previous automatic configuration selected, without that Hub lookup.
    config.dtype = torch.bfloat16
    return config


def patch_transformers_for_qwen_asr():
    """Let qwen-asr 0.0.6 import under transformers 5, whose decorator takes the function directly.

    Only needed on Windows, where the community vLLM build requires transformers>=5
    while qwen-asr pins 4.57.6. Its HF backend is imported but vLLM serves the model.
    """
    from transformers.utils import generic

    original = generic.check_model_inputs
    if getattr(original, "_translate_video_patched", False):
        return

    def check_model_inputs(func=None):
        return original if func is None else original(func)

    check_model_inputs._translate_video_patched = True
    generic.check_model_inputs = check_model_inputs

    # transformers 5 ships a native qwen3_asr config; qwen-asr registers its own again.
    from transformers import AutoConfig, AutoModel, AutoProcessor

    for auto in (AutoConfig, AutoModel, AutoProcessor):
        register = auto.register

        def tolerant(*args, _register=register, **kwargs):
            kwargs.setdefault("exist_ok", True)
            return _register(*args, **kwargs)

        auto.register = tolerant
