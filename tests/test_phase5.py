from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

import dcoupler as dc
from tests.conftest import require_modules


def _load_dodo_module():
    dodo_repo = os.environ.get("DODO_REPO", "/Users/darrieythorsson/compHydro/code/DODO")
    dodo_repo = Path(dodo_repo)
    if not dodo_repo.exists():
        pytest.skip("DODO repo not found; set DODO_REPO to enable Phase 5 test")
    sys.path.insert(0, str(dodo_repo / "python"))
    try:
        import dodo.run_coupled_optimization as dodo
    except Exception as exc:
        pytest.skip(f"Unable to import DODO: {exc}")
    return dodo


@pytest.mark.heavy
def test_three_component_coupling_with_dgw():
    require_modules("cfuse", "droute", "dgw", "xarray", "pandas")
    dodo = _load_dodo_module()

    local_data = Path(__file__).resolve().parent.parent / "data"
    cli_path = None
    if local_data.exists():
        for name in [
            "domain_Bow_at_Banff_distributed",
            "bow_banff_distributed",
            "domain_Bow_at_Banff_semi_distributed",
        ]:
            candidate = local_data / name
            if candidate.exists():
                cli_path = str(candidate)
                break

    data_path = dodo.resolve_data_path(cli_path, domain="distributed")

    try:
        forcing, obs, topo_file, hru_areas, time_vals, n_bands = dodo.load_data_hourly(data_path)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    gw_cfg = dodo.load_groundwater_config(Path(data_path))
    if gw_cfg is None:
        pytest.skip("Groundwater config not found under data_path/groundwater")

    from dodo.groundwater_coupler import (
        load_groundwater_mesh,
        GroundwaterParameterModule,
        DifferentiableGroundwater,
    )

    gw_config, mapper = load_groundwater_mesh(
        gw_cfg["mesh_file"],
        gw_cfg["hru_mapping_file"],
        gw_cfg.get("river_mapping_file"),
    )
    gw_config.dt = 3600.0

    import dgw
    import dgw_py
    import xarray as xr

    has_native_coupler = hasattr(dgw, "GroundwaterModel") and hasattr(dgw, "SolverConfig")

    if has_native_coupler:
        from dodo.groundwater_coupler import GroundwaterCoupler

        if str(gw_cfg["mesh_file"]).endswith(".nc"):
            def _mesh_from_nc(path: str):
                dom = xr.open_dataset(path)
                mesh = dgw_py.build_mesh_from_arrays(
                    dom.cell_centroid_x.values.astype("float64"),
                    dom.cell_centroid_y.values.astype("float64"),
                    dom.cell_z_surface.values.astype("float64"),
                    dom.cell_z_bottom.values.astype("float64"),
                    dom.cell_area.values.astype("float64"),
                    dom.face_cell_left.values.astype("int64"),
                    dom.face_cell_right.values.astype("int64"),
                    dom.face_length.values.astype("float64"),
                    dom.face_distance.values.astype("float64"),
                    dom.face_normal_x.values.astype("float64"),
                    dom.face_normal_y.values.astype("float64"),
                    dom.river_seg_cell.values.astype("int64"),
                    dom.river_seg_length.values.astype("float64"),
                    dom.river_seg_width.values.astype("float64"),
                    dom.river_streambed_elev.values.astype("float64"),
                    dom.river_streambed_K.values.astype("float64"),
                    dom.river_streambed_thickness.values.astype("float64"),
                    dom.river_downstream_idx.values.astype("int64"),
                )
                dom.close()
                return mesh

            dgw.Mesh.from_file = staticmethod(_mesh_from_nc)

        gw_coupler = GroundwaterCoupler(gw_config, mapper)
        gw_params = GroundwaterParameterModule(mapper.n_cells, mapper.n_river_cells, gw_config)
    else:
        gw_coupler = None
        gw_params = GroundwaterParameterModule(mapper.n_cells, mapper.n_river_cells, gw_config)
        if str(gw_cfg["mesh_file"]).endswith(".nc"):
            dom = xr.open_dataset(gw_cfg["mesh_file"])
            _ = dgw_py.build_mesh_from_arrays(
                dom.cell_centroid_x.values.astype("float64"),
                dom.cell_centroid_y.values.astype("float64"),
                dom.cell_z_surface.values.astype("float64"),
                dom.cell_z_bottom.values.astype("float64"),
                dom.cell_area.values.astype("float64"),
                dom.face_cell_left.values.astype("int64"),
                dom.face_cell_right.values.astype("int64"),
                dom.face_length.values.astype("float64"),
                dom.face_distance.values.astype("float64"),
                dom.face_normal_x.values.astype("float64"),
                dom.face_normal_y.values.astype("float64"),
                dom.river_seg_cell.values.astype("int64"),
                dom.river_seg_length.values.astype("float64"),
                dom.river_seg_width.values.astype("float64"),
                dom.river_streambed_elev.values.astype("float64"),
                dom.river_streambed_K.values.astype("float64"),
                dom.river_streambed_thickness.values.astype("float64"),
                dom.river_downstream_idx.values.astype("int64"),
            )
            dom.close()

    class GroundwaterComponent(dc.DifferentiableComponent, torch.nn.Module):
        def __init__(self, name: str, n_reaches: int, reach_id_to_idx: dict) -> None:
            super().__init__()
            self._name = name
            self.n_reaches = n_reaches
            self.reach_id_to_idx = reach_id_to_idx

        @property
        def name(self) -> str:
            return self._name

        @property
        def input_fluxes(self):
            return [
                dc.FluxSpec(
                    name="drainage",
                    units="mm/day",
                    direction=dc.FluxDirection.INPUT,
                    spatial_type="hru",
                    temporal_resolution=gw_config.dt,
                    dims=("time", "hru"),
                )
            ]

        @property
        def output_fluxes(self):
            return [
                dc.FluxSpec(
                    name="gw_lateral",
                    units="m3/s",
                    direction=dc.FluxDirection.OUTPUT,
                    spatial_type="reach",
                    temporal_resolution=gw_config.dt,
                    dims=("time", "reach"),
                )
            ]

        @property
        def parameters(self):
            return [
                dc.ParameterSpec(
                    name="K",
                    lower_bound=gw_config.K_min,
                    upper_bound=gw_config.K_max,
                    spatial=True,
                    n_spatial=mapper.n_cells,
                    log_transform=True,
                ),
                dc.ParameterSpec(
                    name="Sy",
                    lower_bound=gw_config.Sy_min,
                    upper_bound=gw_config.Sy_max,
                    spatial=True,
                    n_spatial=mapper.n_cells,
                    log_transform=False,
                ),
                dc.ParameterSpec(
                    name="streambed_K",
                    lower_bound=gw_config.streambed_K_min,
                    upper_bound=gw_config.streambed_K_max,
                    spatial=True,
                    n_spatial=mapper.n_river_cells,
                    log_transform=True,
                ),
            ]

        @property
        def gradient_method(self):
            return dc.GradientMethod.ENZYME

        @property
        def state_size(self):
            return 0

        @property
        def requires_batch(self):
            return True

        def get_initial_state(self):
            return torch.empty(0)

        def step(self, inputs, state, dt):
            raise RuntimeError("GroundwaterComponent requires batch execution")

        def run(self, inputs, state, dt, n_timesteps):
            drainage = inputs.get("drainage")
            if drainage is None:
                raise ValueError("Missing required input 'drainage'")

            K, Sy, streambed_K = gw_params()
            if gw_coupler is not None:
                exchange = DifferentiableGroundwater.apply(
                    drainage,
                    K,
                    Sy,
                    streambed_K,
                    gw_coupler,
                    None,
                )
            else:
                # Fallback: simple differentiable proxy using parameters
                scale = torch.sigmoid(K.mean()) + torch.sigmoid(Sy.mean())
                exchange = drainage.mean(dim=1, keepdim=True) * scale
                exchange = exchange.repeat(1, mapper.n_river_cells) if mapper.n_river_cells > 0 else exchange

            gw_lateral = torch.zeros(
                exchange.shape[0],
                self.n_reaches,
                device=exchange.device,
            )

            if mapper.n_river_cells > 0:
                for i, (_, reach_id) in enumerate(mapper.river_cell_to_reach.items()):
                    reach_idx = self.reach_id_to_idx.get(int(reach_id))
                    if reach_idx is not None:
                        gw_lateral[:, reach_idx] += exchange[:, i]

            return {"gw_lateral": gw_lateral}, state

        def get_torch_parameters(self):
            return list(gw_params.parameters())

        def get_physical_parameters(self):
            K, Sy, streambed_K = gw_params()
            return {"K": K, "Sy": Sy, "streambed_K": streambed_K}

    n_steps = 8
    dt_seconds = 3600.0
    forcing = forcing[:n_steps]

    land = dc.FUSEComponent(
        name="land",
        fuse_config=dodo.cfuse.VIC_CONFIG,
        n_hrus=len(hru_areas),
        dt=dt_seconds,
        spatial_params=True,
    )

    routing = dc.MuskingumCungeRouting(
        name="routing",
        topology_file=topo_file,
        hru_areas=hru_areas,
        dt=dt_seconds,
    )

    gw = GroundwaterComponent("groundwater", routing.n_reaches, routing.id_to_idx)

    graph = dc.CouplingGraph()
    graph.add_component(land)
    graph.add_component(gw)
    graph.add_component(routing)

    remapper = dc.SpatialRemapper.from_sparse(routing.mapping_matrix)

    graph.connect(
        "land",
        "runoff",
        "routing",
        "lateral_inflow",
        spatial_remap=remapper,
        unit_conversion=1.0,
    )
    graph.connect(
        "land",
        "runoff",
        "groundwater",
        "drainage",
    )
    graph.connect(
        "groundwater",
        "gw_lateral",
        "routing",
        "lateral_inflow",
    )

    outputs = graph.forward(
        external_inputs={"land": {"forcing": forcing}},
        initial_states={"land": land.get_initial_state()},
        n_timesteps=n_steps,
        dt=dt_seconds,
    )

    assert "discharge" in outputs["routing"]
    assert outputs["routing"]["discharge"].shape[0] == n_steps
