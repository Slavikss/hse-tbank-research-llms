"""Expression generation and deterministic evaluation for modular arithmetic tasks."""

from __future__ import annotations

import ast
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

_ALLOWED_OPERATORS = {"+", "-", "*"}
_MAX_EXPRESSION_CHARS = 180


@dataclass(frozen=True)
class ArithmeticConfig:
    n_terms: int
    abs_max: int
    operators: tuple[str, ...]
    max_parentheses_depth: int
    modulus: int
    allow_negative_literals: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExpressionSample:
    expression: str
    raw_value: int


_DIFFICULTY_MAP: dict[int, dict[str, object]] = {
    1: {
        "n_terms": 2,
        "abs_max": 20,
        "operators": ("+", "-"),
        "max_parentheses_depth": 0,
        "modulus": 7,
    },
    2: {
        "n_terms": 3,
        "abs_max": 30,
        "operators": ("+", "-"),
        "max_parentheses_depth": 0,
        "modulus": 11,
    },
    3: {
        "n_terms": 3,
        "abs_max": 50,
        "operators": ("+", "-", "*"),
        "max_parentheses_depth": 0,
        "modulus": 13,
    },
    4: {
        "n_terms": 4,
        "abs_max": 75,
        "operators": ("+", "-", "*"),
        "max_parentheses_depth": 1,
        "modulus": 17,
    },
    5: {
        "n_terms": 5,
        "abs_max": 100,
        "operators": ("+", "-", "*"),
        "max_parentheses_depth": 1,
        "modulus": 19,
    },
    6: {
        "n_terms": 6,
        "abs_max": 150,
        "operators": ("+", "-", "*"),
        "max_parentheses_depth": 1,
        "modulus": 23,
    },
    7: {
        "n_terms": 6,
        "abs_max": 200,
        "operators": ("+", "-", "*"),
        "max_parentheses_depth": 2,
        "modulus": 29,
    },
    8: {
        "n_terms": 7,
        "abs_max": 300,
        "operators": ("+", "-", "*"),
        "max_parentheses_depth": 2,
        "modulus": 31,
    },
    9: {
        "n_terms": 8,
        "abs_max": 400,
        "operators": ("+", "-", "*"),
        "max_parentheses_depth": 2,
        "modulus": 37,
    },
    10: {
        "n_terms": 9,
        "abs_max": 500,
        "operators": ("+", "-", "*"),
        "max_parentheses_depth": 3,
        "modulus": 43,
    },
}


def resolve_config(difficulty: int | None, **overrides: object) -> ArithmeticConfig:
    """Resolve difficulty config and apply direct hyper-parameter overrides."""
    if difficulty is None:
        difficulty = 1
    if difficulty not in _DIFFICULTY_MAP:
        raise ValueError(f"difficulty must be in [1, 10], got {difficulty}")

    base = dict(_DIFFICULTY_MAP[difficulty])
    normalized: dict[str, object] = {}

    for key in (
        "n_terms",
        "abs_max",
        "operators",
        "max_parentheses_depth",
        "modulus",
        "allow_negative_literals",
    ):
        if key in overrides and overrides[key] is not None:
            normalized[key] = overrides[key]

    if "operators" in normalized:
        normalized["operators"] = _normalize_operators(normalized["operators"])
    else:
        base["operators"] = tuple(base["operators"])

    merged = {**base, **normalized}

    config = ArithmeticConfig(
        n_terms=int(merged["n_terms"]),
        abs_max=int(merged["abs_max"]),
        operators=tuple(merged["operators"]),
        max_parentheses_depth=int(merged["max_parentheses_depth"]),
        modulus=int(merged["modulus"]),
        allow_negative_literals=bool(merged.get("allow_negative_literals", True)),
    )
    _validate_config(config)
    return config


def generate_expression(
    rng: random.Random,
    config: ArithmeticConfig,
    max_attempts: int,
) -> ExpressionSample:
    """Generate a valid expression under constraints."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0")

    for _ in range(max_attempts):
        try:
            if config.max_parentheses_depth == 0:
                sample = _generate_flat_expression(rng, config)
            else:
                sample = _generate_parenthesized_expression(rng, config)
        except ValueError:
            continue

        if len(sample.expression) > _MAX_EXPRESSION_CHARS:
            continue
        return sample

    raise ValueError(
        f"Failed to generate valid expression after {max_attempts} attempts for config={config}"
    )


def _normalize_operators(raw_ops: object) -> tuple[str, ...]:
    if isinstance(raw_ops, str):
        candidates = [part.strip() for part in raw_ops.split(",") if part.strip()]
    elif isinstance(raw_ops, Iterable):
        candidates = [str(x).strip() for x in raw_ops if str(x).strip()]
    else:
        raise ValueError("operators must be a comma-separated string or iterable")

    unique: list[str] = []
    for op in candidates:
        if op not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported operator: {op}")
        if op not in unique:
            unique.append(op)
    if not unique:
        raise ValueError("operators must not be empty")
    return tuple(unique)


def _validate_config(config: ArithmeticConfig) -> None:
    if config.n_terms < 2:
        raise ValueError("n_terms must be >= 2")
    if config.abs_max < 1:
        raise ValueError("abs_max must be >= 1")
    if config.max_parentheses_depth < 0:
        raise ValueError("max_parentheses_depth must be >= 0")
    if config.modulus < 2:
        raise ValueError("modulus must be >= 2")
    if not config.operators:
        raise ValueError("operators must not be empty")
    for op in config.operators:
        if op not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported operator: {op}")


def _sample_literal(rng: random.Random, abs_max: int, allow_negative: bool) -> int:
    if allow_negative:
        value = rng.randint(-abs_max, abs_max)
        if value == 0:
            return 1
        return value
    return rng.randint(1, abs_max)


def _literal_to_text(value: int) -> str:
    return str(value)


def _generate_flat_expression(rng: random.Random, config: ArithmeticConfig) -> ExpressionSample:
    values = [
        _sample_literal(rng, config.abs_max, config.allow_negative_literals)
        for _ in range(config.n_terms)
    ]
    ops = [rng.choice(config.operators) for _ in range(config.n_terms - 1)]
    expression = _flat_to_string(values, ops)
    raw_value = _safe_eval_expression(expression)
    return ExpressionSample(expression=expression, raw_value=raw_value)


def _generate_parenthesized_expression(
    rng: random.Random, config: ArithmeticConfig
) -> ExpressionSample:
    values = [
        _sample_literal(rng, config.abs_max, config.allow_negative_literals)
        for _ in range(config.n_terms)
    ]
    ops = [rng.choice(config.operators) for _ in range(config.n_terms - 1)]

    expression = _build_segment(
        values=values,
        ops=ops,
        start=0,
        end=config.n_terms - 1,
        depth_remaining=config.max_parentheses_depth,
        rng=rng,
        force_wrap=True,
    )
    if _measure_parentheses_depth(expression) > config.max_parentheses_depth:
        raise ValueError("parentheses depth exceeded")
    raw_value = _safe_eval_expression(expression)
    return ExpressionSample(expression=expression, raw_value=raw_value)


def _build_segment(
    values: Sequence[int],
    ops: Sequence[str],
    start: int,
    end: int,
    depth_remaining: int,
    rng: random.Random,
    force_wrap: bool,
) -> str:
    if start == end:
        return _literal_to_text(values[start])

    if depth_remaining <= 0:
        return _flat_to_string(values[start : end + 1], ops[start:end])

    split = rng.randint(start, end - 1)
    left = _build_segment(
        values=values,
        ops=ops,
        start=start,
        end=split,
        depth_remaining=depth_remaining - 1,
        rng=rng,
        force_wrap=False,
    )
    right = _build_segment(
        values=values,
        ops=ops,
        start=split + 1,
        end=end,
        depth_remaining=depth_remaining - 1,
        rng=rng,
        force_wrap=False,
    )
    body = f"{left} {ops[split]} {right}"

    if force_wrap or rng.random() < 0.8:
        return f"({body})"
    return body


def _flat_to_string(values: Sequence[int], ops: Sequence[str]) -> str:
    chunks = [_literal_to_text(values[0])]
    for op, value in zip(ops, values[1:], strict=True):
        chunks.append(op)
        chunks.append(_literal_to_text(value))
    return " ".join(chunks)


def _measure_parentheses_depth(expression: str) -> int:
    depth = 0
    max_depth = 0
    for ch in expression:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced parentheses")
    if depth != 0:
        raise ValueError("Unbalanced parentheses")
    return max_depth


def _safe_eval_expression(expression: str) -> int:
    tree = ast.parse(expression, mode="eval")

    def eval_node(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, int):
                raise ValueError("Only integer literals are allowed")
            return int(node.value)
        if isinstance(node, ast.UnaryOp):
            value = eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            raise ValueError("Unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            raise ValueError("Unsupported binary operator")
        raise ValueError("Unsupported expression node")

    return eval_node(tree)
