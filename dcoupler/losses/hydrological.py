from __future__ import annotations

from typing import Optional
import torch


def _apply_mask(sim: torch.Tensor, obs: torch.Tensor, mask: Optional[torch.Tensor]):
    if mask is None:
        mask = ~torch.isnan(obs)
    return sim[mask], obs[mask]


def nse_loss(sim: torch.Tensor, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Nash-Sutcliffe Efficiency loss (1 - NSE)."""
    sim_v, obs_v = _apply_mask(sim, obs, mask)
    if len(obs_v) == 0:
        return torch.tensor(float("inf"))
    ss_res = torch.sum((sim_v - obs_v) ** 2)
    ss_tot = torch.sum((obs_v - obs_v.mean()) ** 2)
    return ss_res / (ss_tot + 1e-10)


def log_nse_loss(
    sim: torch.Tensor,
    obs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """NSE on log-transformed flows."""
    if mask is None:
        mask = ~torch.isnan(obs) & (obs > 0) & (sim > 0)
    sim_v, obs_v = _apply_mask(sim, obs, mask)
    if len(obs_v) == 0:
        return torch.tensor(float("inf"))
    log_sim = torch.log(sim_v + epsilon)
    log_obs = torch.log(obs_v + epsilon)
    ss_res = torch.sum((log_sim - log_obs) ** 2)
    ss_tot = torch.sum((log_obs - log_obs.mean()) ** 2)
    return ss_res / (ss_tot + 1e-10)


def combined_nse_loss(
    sim: torch.Tensor,
    obs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
) -> torch.Tensor:
    nse = nse_loss(sim, obs, mask)
    log_nse = log_nse_loss(sim, obs, mask)
    return alpha * nse + (1 - alpha) * log_nse


def kge_loss(sim: torch.Tensor, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Kling-Gupta Efficiency loss (1 - KGE)."""
    sim_v, obs_v = _apply_mask(sim, obs, mask)
    if len(obs_v) < 2:
        return torch.tensor(float("inf"))
    r = torch.corrcoef(torch.stack([sim_v, obs_v]))[0, 1]
    alpha = sim_v.std() / (obs_v.std() + 1e-10)
    beta = sim_v.mean() / (obs_v.mean() + 1e-10)
    return torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def flow_duration_loss(sim: torch.Tensor, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Flow duration curve loss."""
    sim_v, obs_v = _apply_mask(sim, obs, mask)
    if len(obs_v) == 0:
        return torch.tensor(float("inf"))
    sim_sorted = torch.sort(sim_v, descending=True)[0]
    obs_sorted = torch.sort(obs_v, descending=True)[0]
    rel_err = (sim_sorted - obs_sorted) / (obs_sorted + 1e-6)
    return torch.mean(rel_err ** 2)


def asymmetric_nse_loss(
    sim: torch.Tensor,
    obs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    under_weight: float = 2.0,
) -> torch.Tensor:
    """NSE with asymmetric weighting (penalizes underestimation more)."""
    sim_v, obs_v = _apply_mask(sim, obs, mask)
    if len(obs_v) == 0:
        return torch.tensor(float("inf"))
    errors = sim_v - obs_v
    weights = torch.where(errors < 0, under_weight, 1.0)
    weighted_ss_res = torch.sum(weights * errors ** 2)
    ss_tot = torch.sum((obs_v - obs_v.mean()) ** 2)
    return weighted_ss_res / (ss_tot + 1e-10)


def peak_weighted_nse(
    sim: torch.Tensor,
    obs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    quantile: float = 0.9,
) -> torch.Tensor:
    """NSE with extra weight on high-flow periods."""
    sim_v, obs_v = _apply_mask(sim, obs, mask)
    if len(obs_v) == 0:
        return torch.tensor(float("inf"))
    threshold = torch.quantile(obs_v, quantile)
    weights = torch.where(obs_v > threshold, 3.0, 1.0)
    weighted_ss_res = torch.sum(weights * (sim_v - obs_v) ** 2)
    weighted_ss_tot = torch.sum(weights * (obs_v - obs_v.mean()) ** 2)
    return weighted_ss_res / (weighted_ss_tot + 1e-10)


def triple_objective_loss(
    sim: torch.Tensor,
    obs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    w_nse: float = 0.4,
    w_log: float = 0.3,
    w_peak: float = 0.3,
) -> torch.Tensor:
    nse = nse_loss(sim, obs, mask)
    log_nse = log_nse_loss(sim, obs, mask)
    peak_nse = peak_weighted_nse(sim, obs, mask)
    return w_nse * nse + w_log * log_nse + w_peak * peak_nse


def rmse_loss(sim: torch.Tensor, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    sim_v, obs_v = _apply_mask(sim, obs, mask)
    if len(obs_v) == 0:
        return torch.tensor(float("inf"))
    return torch.sqrt(torch.mean((sim_v - obs_v) ** 2))


def mse_loss(sim: torch.Tensor, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    sim_v, obs_v = _apply_mask(sim, obs, mask)
    if len(obs_v) == 0:
        return torch.tensor(float("inf"))
    return torch.mean((sim_v - obs_v) ** 2)


def _multi_site(metric_fn, sim: torch.Tensor, obs: torch.Tensor, mask: Optional[torch.Tensor], aggregation: str):
    if sim.dim() != 2 or obs.dim() != 2:
        raise ValueError("multi-site losses expect [time, sites] tensors")
    values = []
    for i in range(sim.shape[1]):
        mask_i = mask[:, i] if mask is not None else None
        values.append(metric_fn(sim[:, i], obs[:, i], mask_i))
    stacked = torch.stack(values)
    if aggregation == "mean":
        return stacked.mean()
    if aggregation == "median":
        return stacked.median()
    if aggregation == "sum":
        return stacked.sum()
    raise ValueError(f"Unknown aggregation '{aggregation}'")


def multi_site_nse(sim: torch.Tensor, obs: torch.Tensor, mask: Optional[torch.Tensor] = None, aggregation: str = "mean"):
    return _multi_site(nse_loss, sim, obs, mask, aggregation)


def multi_site_kge(sim: torch.Tensor, obs: torch.Tensor, mask: Optional[torch.Tensor] = None, aggregation: str = "mean"):
    return _multi_site(kge_loss, sim, obs, mask, aggregation)
