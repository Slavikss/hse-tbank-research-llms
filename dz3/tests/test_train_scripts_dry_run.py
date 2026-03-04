from __future__ import annotations

import importlib


def test_train_modules_importable() -> None:
    modules = [
        "src.rl.train_sft",
        "src.rl.train_grpo",
        "src.rl.train_sft_grpo",
        "src.rl.train_srft",
    ]
    for module in modules:
        importlib.import_module(module)
