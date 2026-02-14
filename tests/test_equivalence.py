from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
import torch

from tests.conftest import require_modules


def _load_dodo_module():
    dodo_repo = os.environ.get("DODO_REPO", "/Users/darrieythorsson/compHydro/code/DODO")
    dodo_repo = Path(dodo_repo)
    if not dodo_repo.exists():
        pytest.skip("DODO repo not found; set DODO_REPO to enable equivalence test")
    sys.path.insert(0, str(dodo_repo / "python"))
    try:
        import dodo.run_coupled_optimization as dodo
    except Exception as exc:
        pytest.skip(f"Unable to import DODO: {exc}")
    return dodo


def _has_cfuse_and_droute():
    try:
        import cfuse  # noqa: F401
        import droute  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.heavy
@pytest.mark.skipif(not _has_cfuse_and_droute(), reason="cfuse/droute not available")
def test_fuse_routing_equivalence_small():
    require_modules("cfuse", "droute", "xarray", "pandas")
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

    n_steps = 48
    forcing = forcing[:n_steps]

    DT_SECONDS = 3600.0

    fuse_config = dodo.cfuse.VIC_CONFIG
    dodo_model = dodo.CoupledFUSERoute(
        fuse_config=fuse_config,
        topology_file=topo_file,
        hru_areas=hru_areas,
        dt=DT_SECONDS,
        warmup_steps=0,
        spatial_params=True,
        routing_method="muskingum_cunge",
    )

    from dcoupler.components import FUSEComponent, MuskingumCungeRouting
    from dcoupler.core import CouplingGraph, SpatialRemapper

    land = FUSEComponent(
        name="land",
        fuse_config=fuse_config,
        n_hrus=len(hru_areas),
        dt=DT_SECONDS,
        spatial_params=True,
    )
    routing = MuskingumCungeRouting(
        name="routing",
        topology_file=topo_file,
        hru_areas=hru_areas,
        dt=DT_SECONDS,
        outlet_reach_id=dodo_model.outlet_reach_id,
    )

    # Sync parameters with DODO model
    with torch.no_grad():
        dodo_params = dodo_model.fuse_raw_params.detach().cpu()
        for i, param in enumerate(land.get_torch_parameters()):
            param.copy_(dodo_params[:, i])
        routing.log_manning_n.copy_(dodo_model.log_manning_n.detach().cpu())

    remapper = SpatialRemapper.from_sparse(dodo_model.mapping_matrix.to_sparse())

    graph = CouplingGraph()
    graph.add_component(land)
    graph.add_component(routing)
    graph.connect(
        "land",
        "runoff",
        "routing",
        "lateral_inflow",
        spatial_remap=remapper,
        unit_conversion=1.0,
    )

    initial_state = dodo_model.get_initial_state()

    dodo_Q, _ = dodo_model(forcing, initial_state)

    outputs = graph.forward(
        external_inputs={"land": {"forcing": forcing}},
        initial_states={"land": initial_state},
        n_timesteps=n_steps,
        dt=DT_SECONDS,
    )
    dc_Q = outputs["routing"]["discharge"]

    max_diff = torch.max(torch.abs(dodo_Q - dc_Q)).item()
    assert max_diff < 1e-5
