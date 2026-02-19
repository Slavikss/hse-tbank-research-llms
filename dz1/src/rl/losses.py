from __future__ import annotations

import torch
import torch.nn.functional as func


def compute_policy_loss(log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    return -(log_probs * advantages).mean()


def compute_entropy_bonus(logits: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.Categorical(logits=logits)
    return dist.entropy().mean()


def compute_value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    return func.mse_loss(values, returns)
