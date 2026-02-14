from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

import dcoupler as dc


def _import_dodo():
    dodo_repo = os.environ.get("DODO_REPO")
    if dodo_repo is None:
        raise RuntimeError(
            "DODO_REPO environment variable not set. "
            "Point it to your local DODO repository clone."
        )
    dodo_repo = Path(dodo_repo)
    if not dodo_repo.exists():
        raise RuntimeError(f"DODO repo not found at {dodo_repo}; check DODO_REPO")
    sys.path.insert(0, str(dodo_repo / "python"))
    import dodo.run_coupled_optimization as dodo
    return dodo


def main():
    try:
        dodo = _import_dodo()
    except Exception as exc:
        raise SystemExit(f"Failed to import DODO helpers: {exc}")

    try:
        data_path = dodo.resolve_data_path(None, domain="distributed")
    except Exception as exc:
        raise SystemExit(f"Failed to resolve data path: {exc}")

    forcing, obs, topo_file, hru_areas, time_vals, n_bands = dodo.load_data_hourly(data_path)

    dt_seconds = 3600.0
    n_steps = 48
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

    graph = dc.CouplingGraph()
    graph.add_component(land)
    graph.add_component(routing)
    graph.connect(
        "land",
        "runoff",
        "routing",
        "lateral_inflow",
        spatial_remap=dc.SpatialRemapper.from_sparse(routing.mapping_matrix),
    )

    outputs = graph.forward(
        external_inputs={"land": {"forcing": forcing}},
        initial_states={"land": land.get_initial_state()},
        n_timesteps=n_steps,
        dt=dt_seconds,
    )

    discharge = outputs["routing"]["discharge"]
    print(f"Discharge shape: {discharge.shape}")
    print(f"Discharge mean: {discharge.mean().item():.3f} m3/s")


if __name__ == "__main__":
    main()
