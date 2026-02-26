"""Core data container for generated tasks and model responses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Data:
    """Data class for environment samples."""

    question: str
    answer: str
    difficulty: int = 1
    metadata: dict[str, Any] | None = None
    gpt_response: str = ""

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def to_json(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "gpt_response": self.gpt_response,
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False)

    @classmethod
    def from_json_str(cls, json_str: str) -> "Data":
        json_data = json.loads(json_str)
        return cls.from_json_dict(json_data)

    @classmethod
    def from_json_dict(cls, json_dict: dict[str, Any]) -> "Data":
        instance = cls(
            question=json_dict["question"],
            answer=str(json_dict["answer"]),
            difficulty=int(json_dict.get("difficulty", 1)),
            metadata=json_dict.get("metadata") or {},
        )
        if "gpt_response" in json_dict:
            instance.gpt_response = str(json_dict["gpt_response"])
        return instance

    @classmethod
    def from_jsonl_file(cls, file_path: str | Path) -> list["Data"]:
        path = Path(file_path)
        data_list: list[Data] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                json_data = json.loads(line)
                data_list.append(cls.from_json_dict(json_data))
        return data_list

    @staticmethod
    def to_jsonl_file(data_list: list["Data"], file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in data_list:
                f.write(item.to_json_str())
                f.write("\n")
