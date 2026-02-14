from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np
import xarray as xr


def main():
    repo_root = Path(__file__).resolve().parent.parent
    dodo_domain = repo_root / "data" / "domain_Bow_at_Banff_distributed"
    topo_file = dodo_domain / "settings" / "mizuRoute" / "topology.nc"

    dgw_mesh = Path(
        "/Users/darrieythorsson/compHydro/code/dgw/data/domain_Bow_at_Banff_elevation_49gru/dgw/domain.nc"
    )

    if not topo_file.exists():
        raise FileNotFoundError(f"Missing topology file: {topo_file}")
    if not dgw_mesh.exists():
        raise FileNotFoundError(f"Missing dGW mesh: {dgw_mesh}")

    gw_dir = dodo_domain / "groundwater"
    gw_dir.mkdir(parents=True, exist_ok=True)

    mesh_target = gw_dir / "mesh.nc"
    shutil.copy2(dgw_mesh, mesh_target)

    topo = xr.open_dataset(topo_file)
    hru_ids = topo["hruId"].values.astype(np.int64)
    hru_areas = topo["area"].values.astype(np.float64)
    seg_ids = topo["segId"].values.astype(np.int64)
    topo.close()

    mesh = xr.open_dataset(mesh_target)
    cell_areas = mesh["cell_area"].values.astype(np.float64)
    n_cells = cell_areas.shape[0]
    mesh.close()

    n_hrus = hru_ids.shape[0]
    cell_ids = np.arange(n_cells, dtype=np.int64)

    weights = np.zeros((n_cells, n_hrus), dtype=np.float64)
    for j in range(n_hrus):
        cell_idx = j % n_cells
        denom = cell_areas[cell_idx] if cell_areas[cell_idx] > 0 else 1.0
        weights[cell_idx, j] = hru_areas[j] / denom

    mapping_ds = xr.Dataset(
        {
            "hru_id": ("hru", hru_ids),
            "cell_id": ("cell", cell_ids),
            "weight": (("cell", "hru"), weights),
            "cell_area": ("cell", cell_areas),
        }
    )
    mapping_path = gw_dir / "hru_to_cell_mapping.nc"
    mapping_ds.to_netcdf(mapping_path)

    n_reaches = seg_ids.shape[0]
    river_cell_ids = cell_ids.copy()
    river_reach_ids = np.array([seg_ids[i % n_reaches] for i in range(n_cells)], dtype=np.int64)

    river_ds = xr.Dataset(
        {
            "cell_id": ("river", river_cell_ids),
            "reach_id": ("river", river_reach_ids),
        }
    )
    river_path = gw_dir / "river_cell_mapping.nc"
    river_ds.to_netcdf(river_path)

    print("Groundwater setup complete:")
    print(f"  mesh: {mesh_target}")
    print(f"  hru_to_cell_mapping: {mapping_path}")
    print(f"  river_cell_mapping: {river_path}")


if __name__ == "__main__":
    main()
