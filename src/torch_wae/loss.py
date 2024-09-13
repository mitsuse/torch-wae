from __future__ import annotations

import torch
from torch.cuda import amp


def n_pair_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    label: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    with amp.autocast(enabled=False):
        simirality = torch.matmul(anchor, positive.t())
        diag = torch.diag(simirality).unsqueeze(1)
        diff = torch.maximum(simirality - diag + margin, torch.zeros_like(simirality))

        m_pos = label.unsqueeze(0) == label.unsqueeze(1)
        m_pos = (1.0 - m_pos.to(diff.dtype)).to(diff.device)
        m_neg = (diff > 0).to(m_pos.dtype)
        m = m_pos * m_neg
        b = label.shape[0]

        loss = torch.sum(m * (torch.exp(diff) - 1.0)) / b

        if torch.isnan(loss).any():
            raise ValueError("Loss is nan")

    return loss
