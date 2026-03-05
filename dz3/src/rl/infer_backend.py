"""Inference backend with vLLM auto-detection and transformers fallback."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def has_vllm() -> bool:
    return _has_module("vllm")


def is_peft_adapter_dir(model_path: str) -> bool:
    path = Path(model_path)
    if not path.is_dir():
        return False
    return (path / "adapter_config.json").exists() and not (path / "config.json").exists()


@dataclass
class SamplingConfig:
    temperature: float
    top_p: float
    max_tokens: int
    seed: int = 2026


class InferenceBackend:
    backend_name: str

    def generate(
        self,
        prompts: Sequence[str],
        n: int,
        sampling: SamplingConfig,
        stop_strings: Sequence[str] | None = None,
    ) -> list[list[str]]:
        raise NotImplementedError


class VLLMBackend(InferenceBackend):
    def __init__(self, model_path: str) -> None:
        from vllm import LLM

        self.backend_name = "vllm"
        self._llm = LLM(model=model_path)

    def generate(
        self,
        prompts: Sequence[str],
        n: int,
        sampling: SamplingConfig,
        stop_strings: Sequence[str] | None = None,
    ) -> list[list[str]]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=n,
            temperature=float(sampling.temperature),
            top_p=float(sampling.top_p),
            max_tokens=int(sampling.max_tokens),
            seed=int(sampling.seed),
            stop=list(stop_strings) if stop_strings else None,
        )
        outputs = self._llm.generate(list(prompts), sampling_params=params)
        generations: list[list[str]] = []
        for out in outputs:
            sample_outputs = [candidate.text for candidate in out.outputs]
            generations.append(sample_outputs)
        return generations


class TransformersBackend(InferenceBackend):
    def __init__(self, model_path: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.backend_name = "transformers"
        self._torch = torch
        load_path = model_path
        if is_peft_adapter_dir(model_path):
            from peft import PeftModel

            adapter_cfg_path = Path(model_path) / "adapter_config.json"
            adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
            base_model_path = str(adapter_cfg.get("base_model_name_or_path", "")).strip()
            if not base_model_path:
                raise RuntimeError(
                    f"PEFT adapter at '{model_path}' is missing base_model_name_or_path"
                )

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                device_map="auto",
            )
            self._model = PeftModel.from_pretrained(base_model, model_path)
            self.backend_name = "transformers-peft"
            load_path = base_model_path
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if self._tokenizer.pad_token is None:
            self._tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()

    def generate(
        self,
        prompts: Sequence[str],
        n: int,
        sampling: SamplingConfig,
        stop_strings: Sequence[str] | None = None,
    ) -> list[list[str]]:
        del stop_strings
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch

        encoded = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=False)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        prompt_len = int(encoded["input_ids"].shape[1])

        all_out: list[list[str]] = [[] for _ in prompts]
        for sample_idx in range(n):
            with torch.no_grad():
                seed_value = int(sampling.seed) + sample_idx
                torch.manual_seed(seed_value)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_value)
                generated = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=float(sampling.temperature),
                    top_p=float(sampling.top_p),
                    max_new_tokens=int(sampling.max_tokens),
                    pad_token_id=tokenizer.pad_token_id,
                )
            continuation = generated[:, prompt_len:]
            decoded = tokenizer.batch_decode(continuation, skip_special_tokens=True)
            for i, text in enumerate(decoded):
                all_out[i].append(text)
        return all_out


def create_backend(model_path: str, backend: str = "auto") -> InferenceBackend:
    normalized = backend.strip().lower()
    if normalized not in {"auto", "vllm", "transformers"}:
        raise ValueError("backend must be one of: auto, vllm, transformers")

    adapter_only = is_peft_adapter_dir(model_path)
    if adapter_only and normalized == "vllm":
        raise RuntimeError(
            "Requested vLLM backend for PEFT adapter-only directory. "
            "Use backend=transformers or save a merged model with config.json."
        )

    if adapter_only and normalized == "auto":
        return TransformersBackend(model_path=model_path)

    if normalized in {"auto", "vllm"} and has_vllm():
        return VLLMBackend(model_path=model_path)

    if normalized == "vllm" and not has_vllm():
        raise RuntimeError("Requested vLLM backend but vllm is not installed")

    return TransformersBackend(model_path=model_path)
