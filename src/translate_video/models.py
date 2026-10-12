"""Real model adapters; heavyweight imports occur only when running inference."""


class QwenASR:
    def __init__(self, model_id: str, language: str, chunk_seconds: float, gpu_fraction: float):
        import os

        # A single ASR stream does not need another full Python engine process.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from qwen_asr import Qwen3ASRModel
        from .vllm_compat import qwen_inference_config, register_qwen_tokenizer

        self.model = Qwen3ASRModel.LLM(
            model=model_id,
            dtype="bfloat16",
            hf_overrides=qwen_inference_config,
            tokenizer_mode=register_qwen_tokenizer(),
            mm_processor_kwargs={"fix_mistral_regex": True},
            gpu_memory_utilization=gpu_fraction,
            max_model_len=1024,
            max_num_batched_tokens=1024,
            max_num_seqs=1,
            swap_space=0,
            max_inference_batch_size=1,
            max_new_tokens=128,
            enforce_eager=True,
        )
        self.language = language
        self.chunk_seconds = chunk_seconds
        self.reset()

    def reset(self):
        self.state = self.model.init_streaming_state(
            language=self.language,
            chunk_size_sec=self.chunk_seconds,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
        )

    def feed(self, audio):
        previous = self.state.chunk_id
        self.model.streaming_transcribe(audio, self.state)
        return self.state.text, self.state.chunk_id != previous

    def finish(self):
        self.model.finish_streaming_transcribe(self.state)
        return self.state.text


class NLLBTranslator:
    def __init__(self, source: str, target: str, device: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.target = target
        model_id = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang=source)
        for code in (source, target):
            if self.tokenizer.convert_tokens_to_ids(code) == self.tokenizer.unk_token_id:
                raise ValueError(f"Unknown NLLB language code: {code}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device).eval()

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        if inputs.input_ids.shape[-1] > 512:
            raise ValueError("Translation segment exceeds the 512-token input limit")
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.target),
                max_new_tokens=256,
                num_beams=1,
            )
        if output[0, -1].item() != self.model.config.eos_token_id:
            raise RuntimeError("Translation reached its token limit without completing")
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]


class MMSTTS:
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoTokenizer, VitsModel

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.is_uroman:
            raise ValueError("This checkpoint requires romanization, which this prototype does not support")
        self.model = VitsModel.from_pretrained(model_id).to(device).eval()
        self.sample_rate = self.model.config.sampling_rate

    def synthesize(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            return self.model(**inputs).waveform[0].cpu().numpy()


class VieNeuTTS:
    """Vietnamese 48 kHz TTS with incremental waveform output."""

    def __init__(self, model_id: str, device: str, voice=None):
        from vieneu import Vieneu

        self.model = Vieneu(
            backbone_repo=model_id, device=device,
            backend="pytorch" if device.startswith("cuda") else "onnx",
            max_streams=1, max_batch_size=1,
        )
        self.sample_rate = self.model.sample_rate
        self.voice = voice

    def synthesize_stream(self, text: str):
        # No watermark extra or reference recording is required for preset voices.
        return self.model.infer_stream(text, voice=self.voice, apply_watermark=False)

    def synthesize(self, text: str):
        import numpy as np

        chunks = list(self.synthesize_stream(text))
        if not chunks:
            raise RuntimeError("VieNeu returned no audio")
        return np.concatenate(chunks)

    def close(self):
        self.model.close()
