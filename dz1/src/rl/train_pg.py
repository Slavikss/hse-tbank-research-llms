from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import PGTrainConfig
from src.envs import make_env, set_global_seeds
from src.models import PolicyMLP, ValueMLP, save_policy_checkpoint
from src.rl.evaluate import evaluate_policy
from src.rl.losses import compute_entropy_bonus, compute_policy_loss, compute_value_loss
from src.rl.returns import flatten_discounted_returns, normalize_advantages, rloo_baselines
from src.utils.logging import ensure_dir, write_csv_rows


@dataclass
class TrainResult:
    run_dir: str
    metrics_csv: str
    summary_json: str
    best_checkpoint: str
    best_eval_reward: float


def _compute_entropy_beta(config: PGTrainConfig, update_idx: int) -> float:
    if config.entropy_schedule == "linear" and config.num_updates > 1:
        progress = update_idx / float(config.num_updates - 1)
        return config.entropy_beta + progress * (config.entropy_beta_end - config.entropy_beta)
    return config.entropy_beta


def _collect_trajectories(
    policy: PolicyMLP,
    env_id: str,
    seed: int,
    trajectories_per_update: int,
    max_steps_per_episode: int,
    device: str,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], int]:
    env = make_env(env_id, seed=seed + seed_offset, max_steps=max_steps_per_episode)
    trajectories: list[dict[str, Any]] = []
    env_steps = 0

    for idx in range(trajectories_per_update):
        obs, _ = env.reset(seed=seed + seed_offset + idx)
        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []

        for _ in range(max_steps_per_episode):
            obs_arr = np.asarray(obs, dtype=np.float32)
            obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)
            logits = policy(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = int(dist.sample().item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            states.append(obs_arr)
            actions.append(action)
            rewards.append(float(reward))
            env_steps += 1
            obs = next_obs

            if terminated or truncated:
                break

        trajectories.append({"states": states, "actions": actions, "rewards": rewards})

    env.close()
    return trajectories, env_steps


def train_policy_gradient(config: PGTrainConfig) -> TrainResult:
    if config.baseline not in {"none", "moving_avg", "value", "rloo"}:
        msg = f"Unsupported baseline: {config.baseline}"
        raise ValueError(msg)

    set_global_seeds(config.seed)
    device = torch.device(config.device)

    obs_dim = 4
    act_dim = 2
    policy = PolicyMLP(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=config.hidden_sizes).to(
        device
    )
    policy_opt = torch.optim.Adam(policy.parameters(), lr=config.lr_policy)

    value_net = None
    value_opt = None
    if config.baseline == "value":
        value_net = ValueMLP(obs_dim=obs_dim, hidden_sizes=config.hidden_sizes).to(device)
        value_opt = torch.optim.Adam(value_net.parameters(), lr=config.lr_value)

    run_dir = Path(config.output_dir) / config.run_name
    ensure_dir(run_dir)
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"
    best_ckpt = run_dir / "best.pt"
    final_ckpt = run_dir / "final.pt"

    moving_avg_baseline: float | None = None
    metrics_rows: list[dict[str, Any]] = []
    best_eval = float("-inf")
    total_env_steps = 0

    for update in range(config.num_updates):
        trajectories, env_steps = _collect_trajectories(
            policy=policy,
            env_id=config.env_id,
            seed=config.seed,
            trajectories_per_update=config.trajectories_per_update,
            max_steps_per_episode=config.max_steps_per_episode,
            device=config.device,
            seed_offset=10_000 * update,
        )
        total_env_steps += env_steps

        episode_returns = [float(np.sum(traj["rewards"])) for traj in trajectories]

        states = np.concatenate(
            [np.asarray(traj["states"], dtype=np.float32) for traj in trajectories],
            axis=0,
        )
        actions = np.concatenate(
            [np.asarray(traj["actions"], dtype=np.int64) for traj in trajectories],
            axis=0,
        )
        returns = flatten_discounted_returns(
            [traj["rewards"] for traj in trajectories],
            gamma=config.gamma,
        )

        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        value_loss_val = 0.0

        baseline_value = np.nan
        advantage_center = True

        if config.baseline == "none":
            advantages = returns.copy()
        elif config.baseline == "moving_avg":
            batch_mean = float(np.mean(episode_returns))
            if moving_avg_baseline is None:
                moving_avg_baseline = batch_mean
            else:
                moving_avg_baseline = (
                    1.0 - config.moving_avg_alpha
                ) * moving_avg_baseline + config.moving_avg_alpha * batch_mean
            baseline_value = float(moving_avg_baseline)
            # Keep baseline effect visible by avoiding the extra mean-centering step.
            advantage_center = False
            advantages = returns - baseline_value
        elif config.baseline == "value":
            assert value_net is not None
            assert value_opt is not None
            value_pred = value_net(states_t).squeeze(-1)
            value_loss = compute_value_loss(value_pred, returns_t)
            value_opt.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), config.grad_clip_norm)
            value_opt.step()
            value_loss_val = float(value_loss.item())
            advantages = (returns_t - value_pred.detach()).cpu().numpy()
        else:
            traj_totals = [float(np.sum(traj["rewards"])) for traj in trajectories]
            traj_baselines = rloo_baselines(traj_totals)
            baseline_per_step: list[float] = []
            for traj_idx, traj in enumerate(trajectories):
                baseline_per_step.extend([traj_baselines[traj_idx]] * len(traj["rewards"]))
            baseline_value = float(np.mean(traj_baselines))
            advantages = returns - np.asarray(baseline_per_step, dtype=np.float32)

        if config.normalize_advantage:
            advantages = normalize_advantages(advantages, center=advantage_center)

        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        logits = policy(states_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = compute_entropy_bonus(logits)

        entropy_beta = _compute_entropy_beta(config, update)
        pg_loss = compute_policy_loss(log_probs=log_probs, advantages=advantages_t)
        total_loss = pg_loss - entropy_beta * entropy

        policy_opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip_norm)
        policy_opt.step()

        row: dict[str, Any] = {
            "update": update,
            "env_steps": total_env_steps,
            "train_return_mean": float(np.mean(episode_returns)),
            "train_return_std": float(np.std(episode_returns)),
            "policy_loss": float(pg_loss.item()),
            "entropy": float(entropy.item()),
            "entropy_beta": float(entropy_beta),
            "value_loss": float(value_loss_val),
            "baseline_value": baseline_value,
            "eval_return_mean": np.nan,
            "eval_return_std": np.nan,
        }

        if update % config.eval_every == 0 or update == config.num_updates - 1:
            eval_stats = evaluate_policy(
                env_id=config.env_id,
                policy=policy,
                episodes=config.eval_episodes,
                seed=config.seed + 500_000 + update,
                max_steps=config.max_steps_per_episode,
                device=config.device,
                deterministic=True,
            )
            row["eval_return_mean"] = float(eval_stats["mean"])
            row["eval_return_std"] = float(eval_stats["std"])

            if eval_stats["mean"] > best_eval:
                best_eval = float(eval_stats["mean"])
                save_policy_checkpoint(
                    path=best_ckpt,
                    policy=policy,
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    hidden_sizes=config.hidden_sizes,
                    extra={
                        "config": asdict(config),
                        "best_eval_reward": best_eval,
                        "total_env_steps": total_env_steps,
                    },
                )

            if eval_stats["mean"] >= config.early_stop_reward:
                metrics_rows.append(row)
                break

        metrics_rows.append(row)

        if total_env_steps >= config.max_train_steps:
            break

    save_policy_checkpoint(
        path=final_ckpt,
        policy=policy,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=config.hidden_sizes,
        extra={
            "config": asdict(config),
            "best_eval_reward": best_eval,
            "total_env_steps": total_env_steps,
        },
    )

    write_csv_rows(metrics_path, metrics_rows)

    summary = {
        "run_name": config.run_name,
        "seed": config.seed,
        "baseline": config.baseline,
        "entropy_beta": config.entropy_beta,
        "entropy_beta_end": config.entropy_beta_end,
        "entropy_schedule": config.entropy_schedule,
        "best_eval_reward": float(best_eval),
        "total_env_steps": int(total_env_steps),
        "metrics_csv": str(metrics_path),
        "best_checkpoint": str(best_ckpt),
        "final_checkpoint": str(final_ckpt),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return TrainResult(
        run_dir=str(run_dir),
        metrics_csv=str(metrics_path),
        summary_json=str(summary_path),
        best_checkpoint=str(best_ckpt),
        best_eval_reward=float(best_eval),
    )
