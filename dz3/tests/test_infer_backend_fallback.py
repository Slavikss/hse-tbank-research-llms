from __future__ import annotations

import json
from pathlib import Path

import src.rl.infer_backend as infer_backend


class _DummyBackend:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.backend_name = "dummy"


def test_auto_falls_back_to_transformers(monkeypatch) -> None:
    monkeypatch.setattr(infer_backend, "has_vllm", lambda: False)
    monkeypatch.setattr(infer_backend, "TransformersBackend", _DummyBackend)

    backend = infer_backend.create_backend(model_path="model", backend="auto")
    assert backend.backend_name == "dummy"


def test_vllm_explicit_without_module_raises(monkeypatch) -> None:
    monkeypatch.setattr(infer_backend, "has_vllm", lambda: False)
    try:
        infer_backend.create_backend(model_path="model", backend="vllm")
    except RuntimeError as exc:
        assert "vLLM" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_detect_peft_adapter_dir(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter_model"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert infer_backend.is_peft_adapter_dir(str(adapter_dir))


def test_auto_uses_transformers_for_peft_adapter_dir(tmp_path: Path, monkeypatch) -> None:
    adapter_dir = tmp_path / "adapter_model"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(infer_backend, "TransformersBackend", _DummyBackend)

    backend = infer_backend.create_backend(model_path=str(adapter_dir), backend="auto")
    assert backend.backend_name == "dummy"


def test_vllm_explicit_for_peft_adapter_dir_raises(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter_model"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}),
        encoding="utf-8",
    )

    try:
        infer_backend.create_backend(model_path=str(adapter_dir), backend="vllm")
    except RuntimeError as exc:
        assert "PEFT adapter-only" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")
