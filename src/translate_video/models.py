"""Real model adapters; heavyweight imports occur only when running inference."""


class QwenASR:
    def __init__(self, model_id: str, language: str | None, chunk_seconds: float, gpu_fraction: float):
        import os

        # A single ASR stream does not need another full Python engine process.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from .vllm_compat import patch_transformers_for_qwen_asr, qwen_inference_config, register_qwen_tokenizer

        patch_transformers_for_qwen_asr()
        from qwen_asr import Qwen3ASRModel

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
            max_inference_batch_size=1,
            max_new_tokens=128,
            enforce_eager=True,
        )
        self.language = language
        self.chunk_seconds = chunk_seconds
        self.reset()

    def set_language(self, language: str | None):
        """None lets Qwen detect the language itself (forcing the wrong one invents text in that language)."""
        self.language = language

    def reset(self):
        self.state = self.model.init_streaming_state(
            language=self.language,
            chunk_size_sec=self.chunk_seconds,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
        )

    def feed(self, audio):
        from .language import clean_asr_text

        previous = self.state.chunk_id
        self.model.streaming_transcribe(audio, self.state)
        return clean_asr_text(self.state.text), self.state.chunk_id != previous

    def finish(self):
        from .language import clean_asr_text

        self.model.finish_streaming_transcribe(self.state)
        return clean_asr_text(self.state.text)


class NLLBTranslator:
    """NLLB-200 translator. The source-language tag follows each sentence's script (`auto_source`), because
    a Chinese sentence tagged `eng_Latn` comes back unchanged or garbled. It never raises for one bad
    sentence: a long-running live session must survive a degenerate generation."""

    def __init__(self, source: str, target: str, device: str, auto_source: bool = True):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.target = target
        self.source = source
        self.auto_source = auto_source
        self.stats = {"retried": 0, "unfinished": 0, "untranslated": 0}
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

    def set_source(self, code: str):
        self.source = code

    def generate(self, inputs, **options):
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.target),
                max_new_tokens=min(256, max(24, int(inputs.input_ids.shape[-1] * 3))),
                no_repeat_ngram_size=4,
                **options,
            )
        finished = output[0, -1].item() in (self.model.config.eos_token_id, self.model.config.pad_token_id)
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip(), finished

    def translate(self, text: str) -> str:
        from .language import contains_cjk, nllb_source_code, prepare_for_translation

        text = prepare_for_translation(text)
        code = nllb_source_code(text, self.source) if self.auto_source else self.source
        self.tokenizer.src_lang = code
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        target_is_cjk = self.target.startswith(("zho", "jpn"))

        def copied(candidate_text):            # CJK left in a non-CJK translation = the source was copied
            return contains_cjk(candidate_text) and not target_is_cjk

        def bad(candidate):
            candidate_text, done = candidate
            return not candidate_text or not done or copied(candidate_text)

        best = self.generate(inputs, num_beams=1)
        if bad(best):
            # Greedy decoding looped, stopped without a translation, or copied the source: try beam search.
            self.stats["retried"] += 1
            retry = self.generate(inputs, num_beams=4, repetition_penalty=1.2)
            options = [best, retry]
            best = (next((c for c in options if not bad(c)), None)
                    or next((c for c in options if c[0] and not copied(c[0])), None)
                    or retry)
        text, done = best
        self.stats["unfinished"] += not done
        self.stats["untranslated"] += copied(text)
        return text


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
        # The SDK captures its CUDA graph lazily on a worker thread. Do it now, before
        # vLLM runs ASR on the main thread, or the two collide ("stream is capturing").
        for _ in self.synthesize_stream("Xin chào."):
            pass

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
