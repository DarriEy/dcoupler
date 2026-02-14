#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import math


def _ensure_python_deps() -> None:
    try:
        import torch  # noqa: F401
        import xarray  # noqa: F401
        import pandas  # noqa: F401
        return
    except Exception:
        pass

    candidates = [
        os.environ.get("SYMFLUENCE_SITE_PACKAGES", ""),
        "/Users/darrieythorsson/compHydro/code/SYMFLUENCE/.venv/lib/python3.11/site-packages",
    ]
    for path in candidates:
        if path and Path(path).exists():
            sys.path.insert(0, path)
            break

    # Re-check after path injection.
    import torch  # noqa: F401
    import xarray  # noqa: F401
    import pandas  # noqa: F401


def _import_dodo():
    dodo_repo = os.environ.get("DODO_REPO", "/Users/darrieythorsson/compHydro/code/DODO")
    dodo_repo = Path(dodo_repo)
    if not dodo_repo.exists():
        raise RuntimeError("DODO repo not found; set DODO_REPO")
    sys.path.insert(0, str(dodo_repo / "python"))
    import dodo.run_coupled_optimization as dodo
    return dodo


def _select_data_path(dodo) -> Path:
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
    return Path(data_path)


def _auto_forcing_scale(data_path: Path) -> float:
    forcing_dir = data_path / "forcing" / "FUSE_input"
    hourly_candidates = sorted(forcing_dir.glob("*_input*.nc"))
    if not hourly_candidates:
        return 1.0
    hourly_file = hourly_candidates[0]
    daily_candidates = sorted(forcing_dir.glob("*_input_daily*.nc"))
    if not daily_candidates:
        return 1.0
    daily_file = daily_candidates[0]

    try:
        import xarray as xr
        import numpy as np
    except Exception:
        return 1.0

    def _mean_var(ds, name: str) -> float:
        var = ds[name]
        vals = var.values.astype("float64")
        fill = var.attrs.get("_FillValue", var.attrs.get("missing_value"))
        if fill is not None:
            vals = np.where(vals == fill, np.nan, vals)
        return float(np.nanmean(vals))

    ds_hourly = xr.open_dataset(hourly_file)
    ds_daily = xr.open_dataset(daily_file)
    try:
        hourly_mean = _mean_var(ds_hourly, "pr")
        daily_mean = _mean_var(ds_daily, "pr")
        units = (ds_hourly["pr"].attrs.get("units", "") or "").lower()
    finally:
        ds_hourly.close()
        ds_daily.close()

    if daily_mean <= 0 or hourly_mean <= 0:
        return 1.0

    ratio = hourly_mean / daily_mean
    if "hour" in units and 0.5 < ratio < 2.0:
        # Hourly file looks like mm/day values; undo the hourly->daily scaling.
        return 1.0 / 24.0
    return 1.0


def _find_fuse_settings(data_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    settings_dir = data_path / "settings" / "FUSE"
    if not settings_dir.exists():
        return None, None
    constraints_files = sorted(settings_dir.glob("fuse_zConstraints*.txt"))
    decisions_files = sorted(settings_dir.glob("fuse_zDecisions*.txt"))
    constraints = constraints_files[0] if constraints_files else None
    decisions = decisions_files[0] if decisions_files else None
    return constraints, decisions


def _load_fuse_config_from_decisions(decisions_file: Path):
    import cfuse

    decisions = cfuse.parse_fuse_decisions(decisions_file)
    config_dict = decisions.to_config_dict()
    config = cfuse.FUSEConfig(**config_dict)
    return config, decisions


_FORTRAN_TO_CFUSE = {
    "MAXWATR_1": "S1_max",
    "MAXWATR_2": "S2_max",
    "FRACTEN": "f_tens",
    "FRCHZNE": "f_rchr",
    "FPRIMQB": "f_base",
    "RTFRAC1": "r1",
    "PERCRTE": "ku",
    "PERCEXP": "c",
    "SACPMLT": "alpha",
    "SACPEXP": "psi",
    "PERCFRAC": "kappa",
    "IFLWRTE": "ki",
    "BASERTE": "ks",
    "QB_POWR": "n",
    "QB_PRMS": "v",
    "QBRATE_2A": "v_A",
    "QBRATE_2B": "v_B",
    "SAREAMAX": "Ac_max",
    "AXV_BEXP": "b",
    "LOGLAMB": "lambda",
    "TISHAPE": "chi",
    "TIMEDELAY": "mu_t",
    "PXTEMP": "T_rain",
    "MBASE": "T_melt",
    "LAPSE": "lapse_rate",
    "OPG": "opg",
    "MFMAX": "MFMAX",
    "MFMIN": "MFMIN",
}


def _fuse_params_from_constraints(constraints_file: Path) -> List[str]:
    params: List[str] = []
    with open(constraints_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("(") or line.startswith("!") or line.startswith("*"):
                continue
            parts = line.split()
            if not parts:
                continue
            fit_flag = parts[0].upper()
            if fit_flag not in {"T", "F"}:
                continue
            if fit_flag != "T":
                continue
            param_name = None
            for part in parts:
                if part in _FORTRAN_TO_CFUSE:
                    param_name = part
                    break
            if not param_name:
                continue
            mapped = _FORTRAN_TO_CFUSE.get(param_name)
            if mapped and mapped not in params:
                params.append(mapped)
    return params


def _build_groundwater_component(
    dodo,
    data_path: Path,
    dt_seconds: float,
    dc,
    torch,
    stage_mode: str,
    stage_value: float,
    init_head: bool,
    init_head_offset: float,
):
    from dodo.groundwater_coupler import (
        load_groundwater_mesh,
        GroundwaterParameterModule,
        DifferentiableGroundwater,
        GroundwaterCoupler,
    )

    gw_cfg = dodo.load_groundwater_config(Path(data_path))
    if gw_cfg is None:
        raise RuntimeError("Groundwater config not found under data_path/groundwater")

    gw_config, mapper = load_groundwater_mesh(
        gw_cfg["mesh_file"],
        gw_cfg["hru_mapping_file"],
        gw_cfg.get("river_mapping_file"),
    )
    gw_config.dt = dt_seconds

    gw_coupler = GroundwaterCoupler(gw_config, mapper)
    gw_params = GroundwaterParameterModule(mapper.n_cells, mapper.n_river_cells, gw_config)
    stage_template = None
    head_init = None

    mesh_file = str(gw_cfg["mesh_file"])
    if mesh_file.lower().endswith(".nc"):
        import xarray as xr

        ds = xr.open_dataset(mesh_file)
        if "cell_z_surface" in ds:
            z_surface = ds["cell_z_surface"].values.astype("float64")
        else:
            z_surface = None
        if "river_streambed_elev" in ds:
            river_bed = ds["river_streambed_elev"].values.astype("float64")
        else:
            river_bed = None
        ds.close()
    else:
        z_surface = None
        river_bed = None

    if mapper.n_river_cells > 0 and stage_mode in {"surface", "streambed"}:
        if stage_mode == "streambed" and river_bed is not None:
            base = river_bed
        else:
            base = z_surface
        if base is not None:
            stage_template = torch.tensor(
                base[mapper.river_cell_ids],
                dtype=torch.float32,
            )

    if init_head:
        if z_surface is not None:
            head_init = z_surface + float(init_head_offset)
        else:
            head_init = None

    if head_init is not None:
        original_reset = gw_coupler.reset_state

        def _reset_and_init():
            original_reset()
            gw_coupler.set_initial_head(head_init)

        gw_coupler.reset_state = _reset_and_init

    class GroundwaterComponent(dc.DifferentiableComponent, torch.nn.Module):
        def __init__(self, name: str, n_reaches: int, reach_id_to_idx: Dict[int, int]) -> None:
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
            stage = None
            if stage_mode in {"zero", "constant"} and mapper.n_river_cells > 0:
                stage = torch.zeros(
                    (drainage.shape[0], mapper.n_river_cells),
                    device=drainage.device,
                    dtype=drainage.dtype,
                )
                if stage_mode == "constant":
                    stage = stage + float(stage_value)
            elif stage_mode in {"surface", "streambed"} and stage_template is not None:
                stage = stage_template.to(drainage.device)
                stage = stage.unsqueeze(0).repeat(drainage.shape[0], 1)
            exchange = DifferentiableGroundwater.apply(
                drainage,
                K,
                Sy,
                streambed_K,
                gw_coupler,
                stage,
            )

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

    return GroundwaterComponent, gw_config


def _metrics(sim, obs, dc) -> Dict[str, float]:
    mask = ~obs.isnan()
    if mask.sum() == 0:
        return {"nse": float("nan"), "rmse": float("nan"), "kge": float("nan")}
    sim = sim[mask]
    obs = obs[mask]
    nse = 1.0 - dc.nse_loss(sim, obs).item()
    rmse = dc.rmse_loss(sim, obs).item()
    kge = 1.0 - dc.kge_loss(sim, obs).item()
    return {"nse": nse, "rmse": rmse, "kge": kge}


def _build_graph(
    dodo,
    data_path,
    forcing,
    topo_file,
    hru_areas,
    dt_seconds,
    use_gw: bool,
    gw_weight: float,
    gw_stage: str,
    gw_stage_value: float,
    init_head: bool,
    init_head_offset: float,
    fuse_shared: bool,
    fuse_config,
    routing_mode: str,
):
    import torch
    import dcoupler as dc

    class RunoffScaler(dc.DifferentiableComponent, torch.nn.Module):
        def __init__(self, name: str) -> None:
            super().__init__()
            self._name = name
            self._min_scale = 1e-3
            self._max_scale = 2.0e3
            self._raw_scale = torch.nn.Parameter(torch.zeros(()))

        @property
        def name(self) -> str:
            return self._name

        @property
        def input_fluxes(self):
            return [
                dc.FluxSpec(
                    name="runoff",
                    units="mm/day",
                    direction=dc.FluxDirection.INPUT,
                    spatial_type="hru",
                    temporal_resolution=dt_seconds,
                    dims=("time", "hru"),
                )
            ]

        @property
        def output_fluxes(self):
            return [
                dc.FluxSpec(
                    name="runoff_scaled",
                    units="mm/day",
                    direction=dc.FluxDirection.OUTPUT,
                    spatial_type="hru",
                    temporal_resolution=dt_seconds,
                    dims=("time", "hru"),
                )
            ]

        @property
        def parameters(self):
            return []

        @property
        def gradient_method(self):
            return dc.GradientMethod.AUTODIFF

        @property
        def state_size(self):
            return 0

        @property
        def requires_batch(self):
            return True

        def get_initial_state(self):
            return torch.empty(0)

        def step(self, inputs, state, dt):
            raise RuntimeError("RunoffScaler requires batch execution")

        def run(self, inputs, state, dt, n_timesteps):
            runoff = inputs.get("runoff")
            if runoff is None:
                raise ValueError("Missing required input 'runoff'")
            scale = self._min_scale + (self._max_scale - self._min_scale) * torch.sigmoid(self._raw_scale)
            return {"runoff_scaled": runoff * scale}, state

        def get_torch_parameters(self):
            return [self._raw_scale]

        def get_physical_parameters(self):
            scale = self._min_scale + (self._max_scale - self._min_scale) * torch.sigmoid(self._raw_scale)
            return {"scale": scale}

    land = dc.FUSEComponent(
        name="land",
        fuse_config=fuse_config,
        n_hrus=len(hru_areas),
        dt=dt_seconds,
        spatial_params=not fuse_shared,
    )

    routing_droute = dc.MuskingumCungeRouting(
        name="routing",
        topology_file=topo_file,
        hru_areas=hru_areas,
        dt=dt_seconds,
    )

    gw = None
    gw_config = None
    if use_gw:
        GroundwaterComponent, gw_config = _build_groundwater_component(
            dodo,
            data_path,
            dt_seconds,
            dc,
            torch,
            gw_stage,
            gw_stage_value,
            init_head,
            init_head_offset,
        )
        gw = GroundwaterComponent("groundwater", routing_droute.n_reaches, routing_droute.id_to_idx)
    scaler = RunoffScaler("scaler")

    graph = dc.CouplingGraph()
    graph.add_component(land)
    graph.add_component(scaler)
    if gw is not None:
        graph.add_component(gw)
    if routing_mode == "sum":
        class OutletSum(dc.DifferentiableComponent, torch.nn.Module):
            def __init__(self, name: str, n_reaches: int, mapping_matrix, id_to_idx):
                super().__init__()
                self._name = name
                self.n_reaches = n_reaches
                self.mapping_matrix = mapping_matrix
                self.id_to_idx = id_to_idx

            @property
            def name(self) -> str:
                return self._name

            @property
            def input_fluxes(self):
                return [
                    dc.FluxSpec(
                        name="lateral_inflow",
                        units="m3/s",
                        direction=dc.FluxDirection.INPUT,
                        spatial_type="reach",
                        temporal_resolution=dt_seconds,
                        dims=("time", "reach"),
                    )
                ]

            @property
            def output_fluxes(self):
                return [
                    dc.FluxSpec(
                        name="discharge",
                        units="m3/s",
                        direction=dc.FluxDirection.OUTPUT,
                        spatial_type="point",
                        temporal_resolution=dt_seconds,
                        dims=("time",),
                    )
                ]

            @property
            def parameters(self):
                return []

            @property
            def gradient_method(self):
                return dc.GradientMethod.AUTOGRAD

            @property
            def state_size(self):
                return 0

            @property
            def requires_batch(self):
                return True

            def get_initial_state(self):
                return torch.empty(0)

            def step(self, inputs, state, dt):
                raise RuntimeError("OutletSum requires batch execution")

            def run(self, inputs, state, dt, n_timesteps):
                lateral = inputs.get("lateral_inflow")
                if lateral is None:
                    raise ValueError("Missing required input 'lateral_inflow'")
                discharge = lateral.sum(dim=1)
                return {"discharge": discharge}, state

            def get_torch_parameters(self):
                return []

            def get_physical_parameters(self):
                return {}

        routing = OutletSum(
            "routing",
            routing_droute.n_reaches,
            routing_droute.mapping_matrix,
            routing_droute.id_to_idx,
        )
    else:
        routing = routing_droute

    graph.add_component(routing)

    remapper = dc.SpatialRemapper.from_sparse(routing.mapping_matrix)

    graph.connect(
        "land",
        "runoff",
        "scaler",
        "runoff",
    )
    graph.connect(
        "scaler",
        "runoff_scaled",
        "routing",
        "lateral_inflow",
        spatial_remap=remapper,
        unit_conversion=1.0,
    )
    if gw is not None:
        graph.connect(
            "scaler",
            "runoff_scaled",
            "groundwater",
            "drainage",
        )
    if gw is not None:
        graph.connect(
            "groundwater",
            "gw_lateral",
            "routing",
            "lateral_inflow",
            unit_conversion=gw_weight,
        )

    return graph, land, routing, gw, scaler, gw_config


def _init_fuse_params_from_defaults(land) -> None:
    import torch
    try:
        import cfuse
    except Exception as exc:
        raise RuntimeError("cfuse is required to initialize defaults") from exc

    defaults = torch.tensor(cfuse.get_default_params_array(), dtype=torch.float32)
    lower = land.param_lower
    upper = land.param_upper
    t = (defaults - lower) / (upper - lower)
    t = torch.clamp(t, 1e-4, 1.0 - 1e-4)
    raw = torch.log(t / (1.0 - t))

    if land.spatial_params:
        for i, param in enumerate(land._raw_params):
            param.data = raw[i].expand(land.n_hrus).clone()
    else:
        for i, param in enumerate(land._raw_params):
            param.data = raw[i].clone()


def _init_fuse_params_from_constraints(
    land,
    constraints_file: Optional[Path],
    decisions,
) -> None:
    import torch
    try:
        import cfuse
        from cfuse.netcdf import parse_fortran_constraints
    except Exception as exc:
        raise RuntimeError("cfuse is required to initialize constraints") from exc

    if constraints_file is None:
        raise RuntimeError("Constraints file not found under settings/FUSE")

    params = parse_fortran_constraints(constraints_file)
    arch2 = getattr(decisions, "arch2", "unlimpow_2")
    values = params.to_cfuse_params(arch2=arch2)
    defaults = cfuse.get_default_params_array()
    if values.size < defaults.size:
        full = defaults.copy()
        full[: values.size] = values
        values = full

    lower = land.param_lower
    upper = land.param_upper
    values_t = torch.tensor(values, dtype=torch.float32)
    t = (values_t - lower) / (upper - lower)
    t = torch.clamp(t, 1e-4, 1.0 - 1e-4)
    raw = torch.log(t / (1.0 - t))

    if land.spatial_params:
        for i, param in enumerate(land._raw_params):
            param.data = raw[i].expand(land.n_hrus).clone()
    else:
        for i, param in enumerate(land._raw_params):
            param.data = raw[i].clone()


def _set_fuse_param_value(land, name: str, value: float) -> None:
    import torch

    try:
        idx = land.param_names.index(name)
    except ValueError as exc:
        raise ValueError(f"Unknown FUSE parameter: {name}") from exc
    lower = float(land.param_lower[idx].item())
    upper = float(land.param_upper[idx].item())
    val = max(min(float(value), upper), lower)
    t = (val - lower) / (upper - lower)
    t = max(min(t, 1.0 - 1e-4), 1e-4)
    raw = torch.log(torch.tensor(t / (1.0 - t)))
    if land.spatial_params:
        land._raw_params[idx].data = raw.expand(land.n_hrus).clone()
    else:
        land._raw_params[idx].data = raw.clone()


def _select_parameters(
    land,
    routing,
    gw,
    scaler,
    optimize: str,
    fuse_param_names: Optional[List[str]] = None,
) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    if optimize in {"fuse_only", "fuse_routing", "all"}:
        if fuse_param_names is None:
            params.extend(land.get_torch_parameters())
        else:
            name_to_idx = {name: i for i, name in enumerate(land.param_names)}
            selected = []
            for name in fuse_param_names:
                idx = name_to_idx.get(name)
                if idx is None:
                    continue
                selected.append(land._raw_params[idx])
            if not selected:
                raise ValueError("No valid FUSE parameters selected for optimization")
            params.extend(selected)
    if optimize in {"fuse_routing", "all"}:
        params.extend(routing.get_torch_parameters())
    if optimize == "all" and gw is not None:
        params.extend(gw.get_torch_parameters())
    params.extend(scaler.get_torch_parameters())
    return params


def run_experiment(
    total_steps: int,
    start: int,
    dt_seconds: float,
    spinup_steps: int,
    train_steps: int,
    eval_steps: int,
    epochs: int,
    lr: float,
    optimize: str,
    loss_name: str,
    loss_alpha: float,
    clip_grad: float,
    eval_every: int,
    patience: int,
    fuse_init: str,
    forcing_scale: Optional[float],
    fuse_shared: bool,
    fuse_config_source: str,
    fuse_params: str,
    routing_mode: str,
    fuse_mu_t: Optional[float],
    use_gw: bool,
    gw_weight: float,
    gw_stage: str,
    gw_stage_value: float,
    init_head: bool,
    init_head_offset: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, float], float]:
    _ensure_python_deps()
    repo_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_dir))
    droute_repo = Path(os.environ.get("DROUTE_REPO", "/Users/darrieythorsson/compHydro/code/dRoute"))
    cfuse_repo = Path(os.environ.get("CFUSE_REPO", "/Users/darrieythorsson/compHydro/code/dFUSE"))
    dgw_repo = Path(os.environ.get("DGW_REPO", "/Users/darrieythorsson/compHydro/code/dgw"))
    dgw_build = Path(os.environ.get("DGW_BUILD", ""))
    if dgw_build and dgw_build.exists():
        sys.path.insert(0, str(dgw_build))
    for path in [droute_repo / "python", cfuse_repo / "python", dgw_repo / "python"]:
        if path.exists():
            sys.path.insert(0, str(path))
    import torch
    import dcoupler as dc

    dodo = _import_dodo()
    data_path = _select_data_path(dodo)
    constraints_file, decisions_file = _find_fuse_settings(data_path)

    decisions = None
    if fuse_config_source == "decisions":
        if decisions_file is None:
            raise RuntimeError("FUSE decisions file not found under settings/FUSE")
        fuse_config, decisions = _load_fuse_config_from_decisions(decisions_file)
    else:
        fuse_config = dodo.cfuse.VIC_CONFIG

    forcing, obs, topo_file, hru_areas, time_vals, n_bands = dodo.load_data_hourly(data_path)
    if forcing_scale is None:
        forcing_scale = _auto_forcing_scale(data_path)
    if forcing_scale != 1.0:
        forcing = forcing * float(forcing_scale)
        print(f"  Applied forcing scale: {forcing_scale:.6f}")
        try:
            precip_mean = forcing[:, :, 0].mean().item()
            pet_mean = forcing[:, :, 1].mean().item()
            print(f"  Scaled forcing means: precip={precip_mean:.4f} mm/day, pet={pet_mean:.4f} mm/day")
        except Exception:
            pass

    end = min(start + total_steps, forcing.shape[0])
    forcing = forcing[start:end]
    obs = obs[start:end]

    if eval_steps == 0:
        eval_steps = max(0, total_steps - spinup_steps - train_steps)

    if spinup_steps + train_steps + eval_steps > forcing.shape[0]:
        raise ValueError("spinup + train + eval exceeds available forcing length")

    graph, land, routing, gw, scaler, gw_config = _build_graph(
        dodo,
        data_path,
        forcing,
        topo_file,
        hru_areas,
        dt_seconds,
        use_gw,
        gw_weight,
        gw_stage,
        gw_stage_value,
        init_head,
        init_head_offset,
        fuse_shared,
        fuse_config,
        routing_mode,
    )
    if fuse_init == "defaults":
        _init_fuse_params_from_defaults(land)
    elif fuse_init == "constraints":
        _init_fuse_params_from_constraints(land, constraints_file, decisions)
    if fuse_mu_t is not None:
        _set_fuse_param_value(land, "mu_t", fuse_mu_t)
    fuse_param_names: Optional[List[str]] = None
    if fuse_params != "all":
        if fuse_params == "constraints":
            if constraints_file is None:
                raise RuntimeError("Constraints file not found for fuse_params=constraints")
            fuse_param_names = _fuse_params_from_constraints(constraints_file)
        else:
            fuse_param_names = [p.strip() for p in fuse_params.split(",") if p.strip()]
    if fuse_mu_t is not None and fuse_param_names:
        if "mu_t" in fuse_param_names:
            fuse_param_names = [p for p in fuse_param_names if p != "mu_t"]
            print("  Fixed mu_t; removed from optimized FUSE params.")
    if fuse_param_names:
        print(f"  FUSE params optimized: {', '.join(fuse_param_names)}")

    params = _select_parameters(land, routing, gw, scaler, optimize, fuse_param_names)
    if not params:
        raise ValueError(f"No parameters selected for optimize={optimize}")

    train_end = spinup_steps + train_steps
    optimizer = torch.optim.Adam(params, lr=lr)

    def _loss(sim, obs):
        if loss_name == "rmse":
            return dc.rmse_loss(sim, obs)
        if loss_name == "nse":
            return dc.nse_loss(sim, obs)
        if loss_name == "log_nse":
            return dc.log_nse_loss(sim, obs)
        if loss_name == "combined":
            return dc.combined_nse_loss(sim, obs, alpha=loss_alpha)
        if loss_name == "kge":
            return dc.kge_loss(sim, obs)
        raise ValueError(f"Unknown loss {loss_name}")

    # Initialize runoff scaler to roughly match mean flow in training window.
    with torch.no_grad():
        # Start from scale = 1.0 to estimate baseline mean.
        base_scale = 1.0
        t0 = (base_scale - scaler._min_scale) / (scaler._max_scale - scaler._min_scale)
        t0 = min(max(t0, 1e-4), 1.0 - 1e-4)
        scaler._raw_scale.data = torch.log(torch.tensor(t0 / (1.0 - t0)))

        outputs = graph.forward(
            external_inputs={"land": {"forcing": forcing[:train_end]}},
            initial_states={"land": land.get_initial_state()},
            n_timesteps=train_end,
            dt=dt_seconds,
        )
        discharge = outputs["routing"]["discharge"]
        train_sim = discharge[spinup_steps:train_end]
        train_obs = obs[spinup_steps:train_end]
        mask = ~train_obs.isnan()
        if mask.any():
            sim_mean = train_sim[mask].mean().item()
            obs_mean = train_obs[mask].mean().item()
            if sim_mean > 0:
                target_scale = max(obs_mean / sim_mean, 1e-3)
                target_scale = max(min(target_scale, scaler._max_scale), scaler._min_scale)
                t = (target_scale - scaler._min_scale) / (scaler._max_scale - scaler._min_scale)
                t = min(max(t, 1e-4), 1.0 - 1e-4)
                scaler._raw_scale.data = torch.log(torch.tensor(t / (1.0 - t)))

    best_eval_loss: Optional[float] = None
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = graph.forward(
            external_inputs={"land": {"forcing": forcing[:train_end]}},
            initial_states={"land": land.get_initial_state()},
            n_timesteps=train_end,
            dt=dt_seconds,
        )
        discharge = outputs["routing"]["discharge"]
        train_obs = obs[:train_end]
        train_sim = discharge[spinup_steps:train_end]
        train_obs = train_obs[spinup_steps:train_end]
        loss = _loss(train_sim, train_obs)
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(params, clip_grad)
        optimizer.step()

        if epoch == 1 or epoch % max(1, epochs // 5) == 0:
            train_metrics = _metrics(train_sim.detach().cpu(), train_obs.detach().cpu(), dc)
            print(
                f"Epoch {epoch}/{epochs} | RMSE {train_metrics['rmse']:.3f} "
                f"| NSE {train_metrics['nse']:.3f} | KGE {train_metrics['kge']:.3f}"
            )

        if eval_every > 0 and epoch % eval_every == 0:
            with torch.no_grad():
                outputs = graph.forward(
                    external_inputs={"land": {"forcing": forcing[: spinup_steps + train_steps + eval_steps]}},
                    initial_states={"land": land.get_initial_state()},
                    n_timesteps=spinup_steps + train_steps + eval_steps,
                    dt=dt_seconds,
                )
                discharge_eval = outputs["routing"]["discharge"].detach().cpu()
            eval_start = train_end
            eval_end = train_end + eval_steps
            eval_sim = discharge_eval[eval_start:eval_end]
            eval_obs = obs[eval_start:eval_end].detach().cpu()
            eval_metrics_now = _metrics(eval_sim, eval_obs, dc)
            eval_loss = _loss(eval_sim, eval_obs).item()
            if best_eval_loss is None or eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_state = {
                    "land": land.state_dict(),
                    "routing": routing.state_dict(),
                    "gw": gw.state_dict() if gw is not None else None,
                    "scaler": scaler.state_dict(),
                }
                no_improve = 0
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    break

    if best_state is not None:
        land.load_state_dict(best_state["land"])
        routing.load_state_dict(best_state["routing"])
        if gw is not None and best_state["gw"] is not None:
            gw.load_state_dict(best_state["gw"])
        scaler.load_state_dict(best_state["scaler"])

    with torch.no_grad():
        outputs = graph.forward(
            external_inputs={"land": {"forcing": forcing[: spinup_steps + train_steps + eval_steps]}},
            initial_states={"land": land.get_initial_state()},
            n_timesteps=spinup_steps + train_steps + eval_steps,
            dt=dt_seconds,
        )
        discharge = outputs["routing"]["discharge"].detach().cpu()
        runoff_mean = float("nan")
        runoff_max = float("nan")
        runoff_scaled_mean = float("nan")
        runoff_scaled_max = float("nan")
        if "land" in outputs and "runoff" in outputs["land"]:
            runoff = outputs["land"]["runoff"].detach()
            runoff_mean = runoff.mean().item()
            runoff_max = runoff.max().item()
        if "scaler" in outputs and "runoff_scaled" in outputs["scaler"]:
            runoff_scaled = outputs["scaler"]["runoff_scaled"].detach()
            runoff_scaled_mean = runoff_scaled.mean().item()
            runoff_scaled_max = runoff_scaled.max().item()
        if "scaler" in outputs:
            runoff_scaled = outputs["scaler"]["runoff_scaled"].detach()
            try:
                runoff_lateral = torch.sparse.mm(
                    routing.mapping_matrix, runoff_scaled.transpose(0, 1)
                ).transpose(0, 1)
                runoff_lateral_mean = runoff_lateral.mean().item()
            except Exception:
                runoff_lateral_mean = float("nan")
        else:
            runoff_lateral_mean = float("nan")
        if gw is not None and "groundwater" in outputs:
            gw_lateral = outputs["groundwater"]["gw_lateral"].detach()
            gw_lateral_mean = gw_lateral.mean().item()
        else:
            gw_lateral_mean = float("nan")

    train_sim = discharge[spinup_steps:train_end]
    train_obs = obs[spinup_steps:train_end].detach().cpu()
    eval_start = train_end
    eval_end = train_end + eval_steps
    eval_sim = discharge[eval_start:eval_end]
    eval_obs = obs[eval_start:eval_end].detach().cpu()

    train_metrics = _metrics(train_sim, train_obs, dc)
    eval_metrics = _metrics(eval_sim, eval_obs, dc)
    scale_val = float(scaler.get_physical_parameters()["scale"].detach().cpu().item())
    if not math.isnan(runoff_lateral_mean):
        print(f"  Mean surface lateral: {runoff_lateral_mean:.6f} m3/s")
    if not math.isnan(gw_lateral_mean):
        print(f"  Mean gw lateral: {gw_lateral_mean:.6f} m3/s (weight {gw_weight})")
    if not math.isnan(runoff_mean):
        print(f"  Mean runoff (raw): {runoff_mean:.3f} mm/day")
        print(f"  Max runoff (raw): {runoff_max:.3f} mm/day")
    if not math.isnan(runoff_scaled_mean):
        print(f"  Mean runoff (scaled): {runoff_scaled_mean:.3f} mm/day")
        print(f"  Max runoff (scaled): {runoff_scaled_max:.3f} mm/day")
    return discharge, obs.detach().cpu(), train_metrics, eval_metrics, scale_val


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 3-component coupling experiment.")
    parser.add_argument("--total-steps", type=int, default=24 * 45, help="Total hourly steps (spinup+train+eval).")
    parser.add_argument("--spinup-steps", type=int, default=24 * 10, help="Spinup steps (ignored in loss).")
    parser.add_argument("--train-steps", type=int, default=24 * 20, help="Training steps after spinup.")
    parser.add_argument("--eval-steps", type=int, default=24 * 15, help="Eval steps after training.")
    parser.add_argument("--start", type=int, default=0, help="Start index in the forcing time series.")
    parser.add_argument("--dt", type=float, default=3600.0, help="Timestep in seconds.")
    parser.add_argument("--epochs", type=int, default=10, help="Calibration epochs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument(
        "--optimize",
        type=str,
        default="fuse_only",
        choices=("fuse_only", "fuse_routing", "all"),
        help="Which parameter groups to optimize.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="combined",
        choices=("rmse", "nse", "log_nse", "combined", "kge"),
        help="Loss function for calibration.",
    )
    parser.add_argument(
        "--loss-alpha",
        type=float,
        default=0.7,
        help="Alpha weight for combined NSE loss.",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Gradient clipping norm (0 to disable).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate on holdout every N epochs (0 to disable).",
    )
    parser.add_argument(
        "--fuse-init",
        type=str,
        default="defaults",
        choices=("defaults", "constraints", "random"),
        help="Initialize FUSE params from cfuse defaults or random values.",
    )
    parser.add_argument(
        "--fuse-config",
        type=str,
        default="vic",
        choices=("vic", "decisions"),
        help="FUSE structural configuration source.",
    )
    parser.add_argument(
        "--fuse-params",
        type=str,
        default="all",
        help="FUSE parameter subset: 'all', 'constraints', or comma-separated names.",
    )
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="droute",
        choices=("droute", "sum"),
        help="Routing mode: droute or simple sum of lateral inflow.",
    )
    parser.add_argument(
        "--fuse-mu-t",
        type=float,
        default=None,
        help="Override FUSE mu_t (time delay, days). Useful to disable internal routing.",
    )
    parser.add_argument(
        "--forcing-scale",
        type=float,
        default=None,
        help="Override forcing scale after loading (default: auto-detect).",
    )
    parser.add_argument(
        "--fuse-shared",
        action="store_true",
        help="Use shared FUSE parameters across HRUs (reduces parameter count).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--no-gw",
        action="store_true",
        help="Disable groundwater component (cfuse + droute only).",
    )
    parser.add_argument(
        "--gw-weight",
        type=float,
        default=1.0,
        help="Scale factor applied to groundwater lateral inflow.",
    )
    parser.add_argument(
        "--gw-stage",
        type=str,
        default="none",
        choices=("none", "zero", "constant", "surface", "streambed"),
        help="Stage forcing for groundwater coupling.",
    )
    parser.add_argument(
        "--gw-stage-value",
        type=float,
        default=1.0,
        help="Constant stage value (m) when gw-stage=constant.",
    )
    parser.add_argument(
        "--init-head",
        action="store_true",
        help="Initialize GW head from mesh surface + offset.",
    )
    parser.add_argument(
        "--init-head-offset",
        type=float,
        default=1.0,
        help="Meters above surface for initial GW head.",
    )
    args = parser.parse_args()

    discharge, obs, train_metrics, eval_metrics, scale_val = run_experiment(
        args.total_steps,
        args.start,
        args.dt,
        args.spinup_steps,
        args.train_steps,
        args.eval_steps,
        args.epochs,
        args.lr,
        args.optimize,
        args.loss,
        args.loss_alpha,
        args.clip_grad,
        args.eval_every,
        args.patience,
        args.fuse_init,
        args.forcing_scale,
        args.fuse_shared,
        args.fuse_config,
        args.fuse_params,
        args.routing_mode,
        args.fuse_mu_t,
        not args.no_gw,
        args.gw_weight,
        args.gw_stage,
        args.gw_stage_value,
        args.init_head,
        args.init_head_offset,
    )

    print("Experiment completed")
    print(f"  Steps: {discharge.shape[0]}")
    print(f"  Discharge mean: {discharge.mean().item():.3f} m3/s")
    print(f"  Discharge max: {discharge.max().item():.3f} m3/s")
    obs_mask = ~obs.isnan()
    print(f"  Obs mean: {obs[obs_mask].mean().item():.3f} m3/s")
    print(f"  Runoff scale: {scale_val:.3f}")
    print(f"  Train NSE: {train_metrics['nse']:.4f}")
    print(f"  Train RMSE: {train_metrics['rmse']:.4f}")
    print(f"  Train KGE: {train_metrics['kge']:.4f}")
    print(f"  Eval NSE: {eval_metrics['nse']:.4f}")
    print(f"  Eval RMSE: {eval_metrics['rmse']:.4f}")
    print(f"  Eval KGE: {eval_metrics['kge']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
