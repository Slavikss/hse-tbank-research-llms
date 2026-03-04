from __future__ import annotations

import torch

from src.rl.srft_objective import compute_srft_loss


def test_srft_loss_components_no_nan() -> None:
    loss = compute_srft_loss(
        sft_ce_loss=torch.tensor(1.2),
        rl_demo_loss=torch.tensor(0.3),
        positive_nll=torch.tensor(0.4),
        negative_logprob=torch.tensor(-0.2),
        entropy=torch.tensor(2.1),
        w_sft_coef=0.5,
        w_rl_coef=0.1,
    )

    assert not torch.isnan(loss.total)
    assert not torch.isnan(loss.sft_demo)
    assert not torch.isnan(loss.rl_demo)
    assert not torch.isnan(loss.rl_self_rollout)
