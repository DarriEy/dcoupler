from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn

from dcoupler.core.component import (
    DifferentiableComponent,
    FluxDirection,
    FluxSpec,
    GradientMethod,
    ParameterSpec,
)

try:
    import cfuse
    import cfuse_core
    from cfuse.torch import DifferentiableFUSEBatch
except ImportError:
    cfuse = None
    cfuse_core = None
    DifferentiableFUSEBatch = None


class FUSEComponent(DifferentiableComponent, nn.Module):
    """FUSE component wrapper using DifferentiableFUSEBatch."""

    def __init__(
        self,
        name: str,
        fuse_config,
        n_hrus: int,
        dt: float = 86400.0,
        spatial_params: bool = True,
        n_states: Optional[int] = None,
    ) -> None:
        super().__init__()
        if DifferentiableFUSEBatch is None or cfuse is None:
            raise RuntimeError("cfuse and DifferentiableFUSEBatch are required for FUSEComponent")
        if fuse_config is None:
            raise ValueError("fuse_config is required")

        self._name = name
        self.fuse_config = fuse_config
        self.config_dict = fuse_config.to_dict() if hasattr(fuse_config, "to_dict") else fuse_config
        self.n_hrus = n_hrus
        self.dt_seconds = float(dt)
        self.dt_days = self.dt_seconds / 86400.0
        self.spatial_params = spatial_params

        if n_states is None:
            if cfuse_core is None:
                raise RuntimeError("cfuse_core required to infer state size")
            self.n_states = int(cfuse_core.get_num_active_states(self.config_dict))
        else:
            self.n_states = int(n_states)

        self.param_names = list(cfuse.PARAM_NAMES)
        self.n_params = len(self.param_names)
        lowers = torch.tensor([cfuse.PARAM_BOUNDS[n][0] for n in self.param_names], dtype=torch.float32)
        uppers = torch.tensor([cfuse.PARAM_BOUNDS[n][1] for n in self.param_names], dtype=torch.float32)
        self.register_buffer("param_lower", lowers)
        self.register_buffer("param_upper", uppers)

        self._raw_params = nn.ParameterList()
        for _ in range(self.n_params):
            if self.spatial_params:
                init = torch.zeros(self.n_hrus) + torch.randn(self.n_hrus) * 0.2
            else:
                init = torch.zeros(())
            self._raw_params.append(nn.Parameter(init))

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec(
                name="forcing",
                units="mixed",
                direction=FluxDirection.INPUT,
                spatial_type="hru",
                temporal_resolution=self.dt_seconds,
                dims=("time", "hru", "var"),
                optional=False,
            )
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec(
                name="runoff",
                units="mm/day",
                direction=FluxDirection.OUTPUT,
                spatial_type="hru",
                temporal_resolution=self.dt_seconds,
                dims=("time", "hru"),
                conserved_quantity="water_mass",
            )
        ]

    @property
    def parameters(self) -> List[ParameterSpec]:
        specs: List[ParameterSpec] = []
        for i, name in enumerate(self.param_names):
            specs.append(
                ParameterSpec(
                    name=name,
                    lower_bound=float(self.param_lower[i].item()),
                    upper_bound=float(self.param_upper[i].item()),
                    spatial=self.spatial_params,
                    n_spatial=self.n_hrus if self.spatial_params else None,
                    log_transform=False,
                )
            )
        return specs

    @property
    def gradient_method(self) -> GradientMethod:
        return GradientMethod.ENZYME

    @property
    def state_size(self) -> int:
        return self.n_states

    @property
    def requires_batch(self) -> bool:
        return True

    def get_initial_state(self) -> torch.Tensor:
        state = torch.zeros(self.n_hrus, self.n_states)
        if self.n_states > 0:
            state[:, 0] = 50.0
        if self.n_states > 1:
            state[:, 1] = 20.0
        if self.n_states > 2:
            state[:, 2] = 200.0
        return state

    def _raw_param_matrix(self) -> torch.Tensor:
        if self.spatial_params:
            return torch.stack(list(self._raw_params), dim=1)
        return torch.stack(list(self._raw_params), dim=0)

    def get_physical_parameters(self) -> Dict[str, torch.Tensor]:
        raw = self._raw_param_matrix()
        if self.spatial_params:
            phys = self.param_lower + (self.param_upper - self.param_lower) * torch.sigmoid(raw)
        else:
            phys = self.param_lower + (self.param_upper - self.param_lower) * torch.sigmoid(raw)
        return {name: phys[:, i] if self.spatial_params else phys[i] for i, name in enumerate(self.param_names)}

    def _physical_param_tensor(self) -> torch.Tensor:
        raw = self._raw_param_matrix()
        if self.spatial_params:
            return self.param_lower + (self.param_upper - self.param_lower) * torch.sigmoid(raw)
        return self.param_lower + (self.param_upper - self.param_lower) * torch.sigmoid(raw)

    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        state: torch.Tensor,
        dt: float,
    ):
        raise RuntimeError("FUSEComponent requires batch execution")

    def run(
        self,
        inputs: Dict[str, torch.Tensor],
        state: torch.Tensor,
        dt: float,
        n_timesteps: int,
    ):
        forcing = inputs.get("forcing")
        if forcing is None:
            raise ValueError("Missing required input 'forcing'")
        phys_params = self._physical_param_tensor()
        if state is None:
            state = self.get_initial_state().to(forcing.device)
        runoff = DifferentiableFUSEBatch.apply(
            phys_params,
            state,
            forcing,
            self.config_dict,
            self.dt_days,
        )
        return {"runoff": runoff}, state

    def get_torch_parameters(self) -> List[torch.nn.Parameter]:
        return list(self._raw_params)
