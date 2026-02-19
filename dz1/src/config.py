from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PGExperimentSpec:
    name: str
    baseline: str
    entropy_beta: float = 0.0
    entropy_beta_end: float = 0.0
    entropy_schedule: str = "constant"


@dataclass
class PGCommonConfig:
    env_id: str = "CartPole-v1"
    gamma: float = 0.99
    lr_policy: float = 3e-4
    lr_value: float = 1e-3
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    trajectories_per_update: int = 10
    num_updates: int = 100
    max_steps_per_episode: int = 500
    normalize_advantage: bool = True
    grad_clip_norm: float = 0.5
    eval_every: int = 5
    eval_episodes: int = 10
    early_stop_reward: float = 495.0
    moving_avg_alpha: float = 0.05
    device: str = "cpu"
    output_dir: str = "artifacts/rl"
    max_train_steps: int = 120_000


@dataclass
class PGTrainConfig(PGCommonConfig):
    seed: int = 0
    run_name: str = "run"
    baseline: str = "none"
    entropy_beta: float = 0.0
    entropy_beta_end: float = 0.0
    entropy_schedule: str = "constant"


@dataclass
class RLSuiteConfig:
    common: PGCommonConfig
    seeds: list[int]
    experiments: list[PGExperimentSpec]


@dataclass
class BCConfig:
    env_id: str = "CartPole-v1"
    seed: int = 0
    device: str = "cpu"
    output_dir: str = "artifacts/bc"
    expert_dataset_episodes: int = 100
    expert_min_reward: float = 450.0
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    epochs: int = 40
    val_ratio: float = 0.2
    early_stop_patience: int = 8
    bc_eval_episodes: int = 20
    coverage_fractions: list[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 1.0])
    shift_perturbation_stds: list[float] = field(default_factory=lambda: [0.0, 0.01, 0.03, 0.05])
    error_compounding_episodes: int = 25
    error_compounding_max_steps: int = 500


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_rl_suite_config(path: str | Path) -> RLSuiteConfig:
    raw = load_yaml(path)
    common = PGCommonConfig(**raw["common"])
    experiments = [PGExperimentSpec(**exp) for exp in raw["experiments"]]
    seeds = list(raw.get("seeds", [0]))
    return RLSuiteConfig(common=common, seeds=seeds, experiments=experiments)


def build_pg_train_config(
    common: PGCommonConfig,
    experiment: PGExperimentSpec,
    seed: int,
) -> PGTrainConfig:
    merged = asdict(common)
    merged.update(
        {
            "seed": seed,
            "run_name": f"{experiment.name}_seed{seed}",
            "baseline": experiment.baseline,
            "entropy_beta": experiment.entropy_beta,
            "entropy_beta_end": experiment.entropy_beta_end,
            "entropy_schedule": experiment.entropy_schedule,
        }
    )
    return PGTrainConfig(**merged)


def load_bc_config(path: str | Path) -> BCConfig:
    raw = load_yaml(path)
    return BCConfig(**raw)
