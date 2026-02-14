from __future__ import annotations

import torch


def interpolate_flux(
    flux: torch.Tensor,
    method: str,
    n_substeps: int,
) -> torch.Tensor:
    """
    Interpolate flux from outer timestep to substeps.

    Args:
        flux: tensor at outer timestep [spatial] or [time, spatial]
        method: "step", "linear", "conservative"
        n_substeps: number of substeps to produce
    Returns:
        tensor with shape [n_substeps, spatial]
    """
    if n_substeps <= 1:
        if flux.dim() == 1:
            return flux.unsqueeze(0)
        return flux

    if flux.dim() == 1:
        base = flux.unsqueeze(0)
    else:
        base = flux

    if method == "step":
        return base.repeat(n_substeps, 1)

    if method == "linear":
        if base.shape[0] < 2:
            return base.repeat(n_substeps, 1)
        start = base[0]
        end = base[1]
        weights = torch.linspace(0, 1, steps=n_substeps, device=base.device).unsqueeze(1)
        return start + (end - start) * weights

    if method == "conservative":
        return base.repeat(n_substeps, 1) / n_substeps

    raise ValueError(f"Unknown interpolation method '{method}'")
