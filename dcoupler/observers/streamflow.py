from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .base import ObservationOperator


class StreamflowObserver(ObservationOperator):
    """Select discharge at specified gauge indices."""

    def __init__(self, gauge_reach_ids: List[int] | None = None, component: str = "routing") -> None:
        self._name = "streamflow"
        self.gauge_reach_ids = gauge_reach_ids or []
        self.component = component

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_model_outputs(self) -> List[Tuple[str, str]]:
        return [(self.component, "discharge")]

    def apply(self, model_outputs: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        discharge = model_outputs[self.component]["discharge"]
        if discharge.dim() == 1:
            return discharge
        if discharge.dim() == 2:
            if not self.gauge_reach_ids:
                return discharge
            return discharge[:, self.gauge_reach_ids]
        raise ValueError("Discharge tensor must be 1D or 2D")
