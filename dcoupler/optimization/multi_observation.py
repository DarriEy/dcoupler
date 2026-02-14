from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from dcoupler.losses import (
    nse_loss,
    log_nse_loss,
    combined_nse_loss,
    kge_loss,
    flow_duration_loss,
    asymmetric_nse_loss,
    peak_weighted_nse,
    triple_objective_loss,
    rmse_loss,
    mse_loss,
)
from dcoupler.observers import ObservationOperator


LOSS_REGISTRY = {
    "nse": nse_loss,
    "log_nse": log_nse_loss,
    "combined": combined_nse_loss,
    "kge": kge_loss,
    "fdc": flow_duration_loss,
    "asymmetric": asymmetric_nse_loss,
    "peak_weighted": peak_weighted_nse,
    "triple": triple_objective_loss,
    "rmse": rmse_loss,
    "mse": mse_loss,
}


@dataclass
class LossTerm:
    observer: ObservationOperator
    observed: torch.Tensor
    loss_fn: callable
    weight: float
    warmup_steps: int
    mask: Optional[torch.Tensor]


class MultiObservationLoss:
    """
    Aggregates loss across multiple observation types and locations.
    """

    def __init__(self) -> None:
        self.terms: List[LossTerm] = []

    def add_term(
        self,
        observer: ObservationOperator,
        observed: torch.Tensor,
        loss_fn: str = "nse",
        weight: float = 1.0,
        warmup_steps: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if loss_fn not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss_fn '{loss_fn}'")
        self.terms.append(
            LossTerm(
                observer=observer,
                observed=observed,
                loss_fn=LOSS_REGISTRY[loss_fn],
                weight=weight,
                warmup_steps=warmup_steps,
                mask=mask,
            )
        )

    def compute(
        self,
        model_outputs: Dict[str, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = torch.tensor(0.0)
        diagnostics: Dict[str, float] = {}

        for idx, term in enumerate(self.terms):
            sim = term.observer.apply(model_outputs)
            obs = term.observed

            if term.warmup_steps > 0 and sim.dim() >= 1:
                sim = sim[term.warmup_steps:]
                obs = obs[term.warmup_steps:]
                mask = term.mask[term.warmup_steps:] if term.mask is not None else None
            else:
                mask = term.mask

            loss_val = term.loss_fn(sim, obs, mask)
            weighted = term.weight * loss_val
            total = total + weighted

            key = f"{term.observer.name}_{idx}"
            diagnostics[key] = float(loss_val.detach().cpu().item())

        return total, diagnostics
