from __future__ import annotations

from typing import Dict, List, Optional

import torch

from dcoupler.core.connection import FluxConnection


class ConservationChecker:
    """
    Verifies and optionally enforces conservation across coupling interfaces.
    """

    def __init__(self, mode: str = "check", tolerance: float = 1e-6):
        if mode not in ("check", "enforce"):
            raise ValueError("mode must be 'check' or 'enforce'")
        self.mode = mode
        self.tolerance = tolerance
        self.conservation_log: List[Dict] = []

    def check_connection(
        self,
        connection: FluxConnection,
        source_flux: torch.Tensor,
        target_flux: torch.Tensor,
        source_areas: torch.Tensor,
        target_areas: torch.Tensor,
        dt: float,
    ) -> Optional[torch.Tensor]:
        """Return corrected target flux if enforcing, else None."""
        if source_flux.numel() == 0 or target_flux.numel() == 0:
            return None

        source_total = torch.sum(source_flux * source_areas) * dt
        target_total = torch.sum(target_flux * target_areas) * dt

        denom = torch.abs(source_total) + 1e-12
        error = torch.abs(source_total - target_total) / denom

        entry = {
            "connection": f"{connection.source_component}.{connection.source_flux} -> "
                         f"{connection.target_component}.{connection.target_flux}",
            "relative_error": float(error.detach().cpu().item()),
        }
        self.conservation_log.append(entry)

        if self.mode == "check":
            return None

        if error < self.tolerance:
            return target_flux

        correction = source_total / (target_total + 1e-12)
        return target_flux * correction
