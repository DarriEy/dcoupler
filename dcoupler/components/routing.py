from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

import droute as dmc

from dcoupler.core.component import (
    DifferentiableComponent,
    FluxDirection,
    FluxSpec,
    GradientMethod,
    ParameterSpec,
)
from dcoupler.wrappers.enzyme import DifferentiableRouting


class MuskingumCungeRouting(DifferentiableComponent, nn.Module):
    """Muskingum-Cunge routing component with Enzyme AD."""

    def __init__(
        self,
        name: str,
        topology_file: str,
        hru_areas: np.ndarray,
        dt: float = 86400.0,
        outlet_reach_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._name = name
        self.dt_seconds = float(dt)

        self.network = self._load_network(topology_file)
        self.n_reaches = self.network.num_reaches()
        self.n_hrus = len(hru_areas)

        topo_order = list(self.network.topological_order())
        self.reach_ids = topo_order
        self.id_to_idx = {rid: i for i, rid in enumerate(topo_order)}

        if outlet_reach_id is None:
            outlet_reach_id = int(topo_order[-1])
        self.outlet_reach_id = int(outlet_reach_id)

        config = dmc.RouterConfig()
        config.dt = self.dt_seconds
        config.num_substeps = 4
        config.enable_gradients = False

        self.router = dmc.MuskingumCungeRouter(self.network, config)

        initial_log_n = torch.full((self.n_reaches,), np.log(0.035))
        initial_log_n = initial_log_n + torch.randn(self.n_reaches) * 0.1
        self.log_manning_n = nn.Parameter(initial_log_n)

        self.register_buffer(
            "mapping_matrix",
            self._build_mapping_matrix(topology_file, hru_areas),
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec(
                name="lateral_inflow",
                units="m3/s",
                direction=FluxDirection.INPUT,
                spatial_type="reach",
                temporal_resolution=self.dt_seconds,
                dims=("time", "reach"),
                optional=False,
            )
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec(
                name="discharge",
                units="m3/s",
                direction=FluxDirection.OUTPUT,
                spatial_type="point",
                temporal_resolution=self.dt_seconds,
                dims=("time",),
                conserved_quantity="water_mass",
            )
        ]

    @property
    def parameters(self) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="manning_n",
                lower_bound=1e-4,
                upper_bound=1.0,
                spatial=True,
                n_spatial=self.n_reaches,
                log_transform=True,
            )
        ]

    @property
    def gradient_method(self) -> GradientMethod:
        return GradientMethod.ENZYME

    @property
    def state_size(self) -> int:
        return 0

    @property
    def requires_batch(self) -> bool:
        return True

    def get_initial_state(self) -> torch.Tensor:
        return torch.empty(0)

    def get_physical_parameters(self) -> Dict[str, torch.Tensor]:
        return {"manning_n": torch.exp(self.log_manning_n)}

    def step(self, inputs: Dict[str, torch.Tensor], state: torch.Tensor, dt: float):
        raise RuntimeError("MuskingumCungeRouting requires batch execution")

    def run(
        self,
        inputs: Dict[str, torch.Tensor],
        state: torch.Tensor,
        dt: float,
        n_timesteps: int,
    ):
        lateral = inputs.get("lateral_inflow")
        if lateral is None:
            raise ValueError("Missing required input 'lateral_inflow'")
        manning_n = torch.exp(self.log_manning_n)
        discharge = DifferentiableRouting.apply(
            lateral,
            manning_n,
            self.router,
            self.network,
            self.outlet_reach_id,
            self.dt_seconds,
        )
        return {"discharge": discharge}, state

    def get_torch_parameters(self) -> List[torch.nn.Parameter]:
        return [self.log_manning_n]

    def _load_network(self, topology_file: str) -> dmc.Network:
        ds = xr.open_dataset(topology_file)

        seg_ids = ds["segId"].values.astype(int)
        down_seg_ids = ds["downSegId"].values.astype(int)
        lengths = ds["length"].values.astype(float)
        slopes = ds["slope"].values.astype(float)
        mann_n = ds["mann_n"].values if "mann_n" in ds else np.full(len(seg_ids), 0.035)

        network = dmc.Network()
        seg_id_set = set(seg_ids)

        upstream_map = {int(sid): [] for sid in seg_ids}
        for i, down_id in enumerate(down_seg_ids):
            if int(down_id) in seg_id_set:
                upstream_map[int(down_id)].append(int(seg_ids[i]))

        for i, sid in enumerate(seg_ids):
            reach = dmc.Reach()
            reach.id = int(sid)
            reach.length = float(lengths[i])
            reach.slope = max(float(slopes[i]), 0.0001)
            reach.manning_n = float(mann_n[i])
            reach.geometry.width_coef = 7.2
            reach.geometry.width_exp = 0.5
            reach.geometry.depth_coef = 0.27
            reach.geometry.depth_exp = 0.3
            reach.upstream_junction_id = int(sid)
            down_id = int(down_seg_ids[i])
            reach.downstream_junction_id = down_id if down_id in seg_id_set else -1
            network.add_reach(reach)

        for i, sid in enumerate(seg_ids):
            junc = dmc.Junction()
            junc.id = int(sid)
            junc.upstream_reach_ids = upstream_map[int(sid)]
            junc.downstream_reach_ids = [int(sid)]
            network.add_junction(junc)

        network.build_topology()
        ds.close()
        return network

    def _build_mapping_matrix(self, topology_file: str, hru_areas: np.ndarray) -> torch.Tensor:
        ds = xr.open_dataset(topology_file)
        hru_to_seg = ds["hruToSegId"].values.astype(int)
        ds.close()

        topo_order = self.network.topological_order()
        id_to_idx = {rid: i for i, rid in enumerate(topo_order)}

        n_hrus = len(hru_to_seg)
        n_reaches = len(topo_order)
        mapping = torch.zeros((n_reaches, n_hrus), dtype=torch.float32)

        for h_idx, seg_id in enumerate(hru_to_seg):
            if seg_id in id_to_idx:
                r_idx = id_to_idx[seg_id]
                conversion = hru_areas[h_idx] / 1000.0 / 86400.0
                mapping[r_idx, h_idx] = conversion

        indices = mapping.nonzero(as_tuple=False).T
        values = mapping[indices[0], indices[1]]
        sparse = torch.sparse_coo_tensor(indices, values, mapping.shape)
        return sparse
