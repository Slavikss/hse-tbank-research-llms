from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as func

from src.config import BCConfig
from src.models import PolicyMLP, save_policy_checkpoint
from src.rl.evaluate import evaluate_policy
from src.utils.logging import ensure_dir, write_csv_rows


@dataclass
class BCTrainResult:
    run_dir: str
    metrics_csv: str
    summary_json: str
    checkpoint_path: str
    best_val_loss: float
    eval_reward_mean: float


def _split_dataset(
    states: np.ndarray,
    actions: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(actions)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    val_n = max(1, int(n * val_ratio))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    if len(train_idx) == 0:
        train_idx = val_idx
    return states[train_idx], actions[train_idx], states[val_idx], actions[val_idx]


def _iter_minibatches(
    states: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(actions))
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start : start + batch_size]
        batches.append((states[batch_idx], actions[batch_idx]))
    return batches


def train_bc_from_arrays(
    states: np.ndarray,
    actions: np.ndarray,
    config: BCConfig,
    run_name: str,
    max_steps: int = 500,
) -> BCTrainResult:
    device = torch.device(config.device)
    run_dir = Path(config.output_dir) / run_name
    ensure_dir(run_dir)

    train_states, train_actions, val_states, val_actions = _split_dataset(
        states=states,
        actions=actions,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    model = PolicyMLP(obs_dim=states.shape[1], act_dim=2, hidden_sizes=config.hidden_sizes).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None
    patience = config.early_stop_patience
    epochs_without_improvement = 0
    metrics: list[dict[str, Any]] = []

    for epoch in range(config.epochs):
        model.train()
        batch_losses: list[float] = []

        for batch_states, batch_actions in _iter_minibatches(
            train_states,
            train_actions,
            batch_size=config.batch_size,
            seed=config.seed + epoch,
        ):
            states_t = torch.tensor(batch_states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(batch_actions, dtype=torch.int64, device=device)
            logits = model(states_t)
            loss = func.cross_entropy(logits, actions_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(val_states, dtype=torch.float32, device=device))
            val_loss = float(
                func.cross_entropy(
                    val_logits,
                    torch.tensor(val_actions, dtype=torch.int64, device=device),
                ).item()
            )

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        metrics.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_stats = evaluate_policy(
        env_id=config.env_id,
        policy=model,
        episodes=config.bc_eval_episodes,
        seed=config.seed + 800_000,
        max_steps=max_steps,
        device=config.device,
        deterministic=True,
    )

    ckpt_path = run_dir / "best.pt"
    save_policy_checkpoint(
        path=ckpt_path,
        policy=model,
        obs_dim=states.shape[1],
        act_dim=2,
        hidden_sizes=config.hidden_sizes,
        extra={
            "config": asdict(config),
            "run_name": run_name,
            "best_val_loss": best_val_loss,
            "eval_reward_mean": float(eval_stats["mean"]),
        },
    )

    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"
    write_csv_rows(metrics_path, metrics)
    summary = {
        "run_name": run_name,
        "checkpoint": str(ckpt_path),
        "best_val_loss": float(best_val_loss),
        "eval_reward_mean": float(eval_stats["mean"]),
        "eval_reward_std": float(eval_stats["std"]),
        "train_samples": int(len(train_actions)),
        "val_samples": int(len(val_actions)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return BCTrainResult(
        run_dir=str(run_dir),
        metrics_csv=str(metrics_path),
        summary_json=str(summary_path),
        checkpoint_path=str(ckpt_path),
        best_val_loss=float(best_val_loss),
        eval_reward_mean=float(eval_stats["mean"]),
    )


def train_bc_from_npz(
    dataset_npz: str | Path,
    config: BCConfig,
    run_name: str = "bc_main",
    max_steps: int = 500,
) -> BCTrainResult:
    data = np.load(Path(dataset_npz))
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.int64)
    return train_bc_from_arrays(states, actions, config, run_name=run_name, max_steps=max_steps)
