from __future__ import annotations

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
