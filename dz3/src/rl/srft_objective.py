"""SRFT loss composition helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SRFTLossComponents:
    total: torch.Tensor
    sft_demo: torch.Tensor
    rl_demo: torch.Tensor
    rl_self_rollout: torch.Tensor
    w_sft: torch.Tensor
    w_rl: torch.Tensor


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return mean token entropy over the last dimension."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean()


def compute_srft_loss(
    sft_ce_loss: torch.Tensor,
    rl_demo_loss: torch.Tensor,
    positive_nll: torch.Tensor,
    negative_logprob: torch.Tensor,
    entropy: torch.Tensor,
    w_sft_coef: float,
    w_rl_coef: float,
) -> SRFTLossComponents:
    """Combine SRFT objective terms using entropy-aware weights."""
    w_sft = float(w_sft_coef) * torch.exp(-entropy.detach())
    w_rl = float(w_rl_coef) * torch.exp(entropy.detach())

    sft_term = w_sft * sft_ce_loss
    rl_self = w_rl * (positive_nll + negative_logprob)
    total = sft_term + rl_demo_loss + rl_self

    return SRFTLossComponents(
        total=total,
        sft_demo=sft_term,
        rl_demo=rl_demo_loss,
        rl_self_rollout=rl_self,
        w_sft=w_sft,
        w_rl=w_rl,
    )
