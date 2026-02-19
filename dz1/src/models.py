from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class PolicyMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> int:
        logits = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        return int(dist.sample().item())


class ValueMLP(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def save_policy_checkpoint(
    path: str | Path,
    policy: PolicyMLP,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: list[int],
    extra: dict | None = None,
) -> None:
    payload = {
        "policy_state_dict": policy.state_dict(),
        "model": {
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden_sizes": hidden_sizes,
        },
    }
    if extra:
        payload.update(extra)
    torch.save(payload, Path(path))


def load_policy_checkpoint(path: str | Path, device: str = "cpu") -> PolicyMLP:
    payload = torch.load(Path(path), map_location=device)
    model_cfg = payload["model"]
    policy = PolicyMLP(
        obs_dim=model_cfg["obs_dim"],
        act_dim=model_cfg["act_dim"],
        hidden_sizes=list(model_cfg["hidden_sizes"]),
    )
    policy.load_state_dict(payload["policy_state_dict"])
    policy.to(device)
    policy.eval()
    return policy
