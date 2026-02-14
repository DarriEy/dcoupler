from __future__ import annotations

from typing import Dict


class TemporalOrchestrator:
    """
    Manages operator splitting when components have different timesteps.
    """

    def __init__(self, outer_dt: float):
        self.outer_dt = float(outer_dt)
        self.component_dt: Dict[str, float] = {}

    def set_component_dt(self, component: str, dt: float) -> None:
        dt = float(dt)
        if self.outer_dt % dt != 0 and dt % self.outer_dt != 0:
            raise ValueError(
                f"Component dt ({dt}s) must evenly divide or be a multiple "
                f"of outer dt ({self.outer_dt}s)"
            )
        self.component_dt[component] = dt

    def get_substeps(self, component: str) -> int:
        comp_dt = self.component_dt.get(component, self.outer_dt)
        if comp_dt <= self.outer_dt:
            return int(self.outer_dt / comp_dt)
        return 1

    def get_component_dt(self, component: str) -> float:
        return self.component_dt.get(component, self.outer_dt)
