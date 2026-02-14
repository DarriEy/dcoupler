from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import os

import torch

logger = logging.getLogger(__name__)

from dcoupler.core.graph import CouplingGraph
from dcoupler.optimization.parameters import ParameterManager
from dcoupler.optimization.multi_observation import MultiObservationLoss


@dataclass
class TrainingResult:
    history: Dict[str, List[float]]
    best_loss: float
    best_epoch: int
    final_parameters: Dict[str, Dict[str, torch.Tensor]]
    convergence_info: Dict


class Trainer:
    """Training loop for coupled model optimization."""

    def __init__(
        self,
        coupling_graph: CouplingGraph,
        loss: MultiObservationLoss,
        param_manager: Optional[ParameterManager] = None,
        optimizer: str = "adam",
        lr: float = 0.01,
        scheduler: str = "warm_restarts",
        grad_clip: float = 1.0,
        n_epochs: int = 100,
        spatial_reg: float = 0.0,
    ) -> None:
        self.graph = coupling_graph
        self.loss = loss
        self.param_manager = param_manager or ParameterManager(coupling_graph)
        self.optimizer_name = optimizer
        self.lr = lr
        self.scheduler_name = scheduler
        self.grad_clip = grad_clip
        self.n_epochs = n_epochs
        self.spatial_reg = spatial_reg

    def _build_optimizer(self):
        params = self.param_manager.get_optimizer_params()
        if not params:
            params = [{"params": self.graph.get_all_parameters()}]

        name = self.optimizer_name.lower()
        if name == "adam":
            return torch.optim.Adam(params, lr=self.lr)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=self.lr)
        if name == "sgd":
            return torch.optim.SGD(params, lr=self.lr, momentum=0.9)
        raise ValueError(f"Unknown optimizer '{self.optimizer_name}'")

    def _build_scheduler(self, optimizer):
        name = self.scheduler_name.lower()
        if name == "none":
            return None, False
        if name == "plateau":
            return (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.7,
                    patience=30,
                    min_lr=1e-6,
                ),
                True,
            )
        if name == "cosine":
            return (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.n_epochs,
                    eta_min=1e-5,
                ),
                False,
            )
        if name == "warm_restarts":
            return (
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=50,
                    T_mult=1,
                    eta_min=1e-5,
                ),
                False,
            )
        if name == "onecycle":
            return (
                torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.lr * 3,
                    total_steps=self.n_epochs,
                    pct_start=0.3,
                    anneal_strategy="cos",
                ),
                False,
            )
        raise ValueError(f"Unknown scheduler '{self.scheduler_name}'")

    def _collect_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        params: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, comp in self.graph.components.items():
            params[name] = comp.get_physical_parameters()
        return params

    def _clone_parameters(self, params: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        cloned: Dict[str, Dict[str, torch.Tensor]] = {}
        for comp_name, comp_params in params.items():
            cloned[comp_name] = {k: v.clone() for k, v in comp_params.items()}
        return cloned

    def train(
        self,
        external_inputs: Dict[str, Dict[str, torch.Tensor]],
        n_timesteps: int,
        dt: float,
        verbose: bool = True,
        checkpoint_every: int = 50,
        checkpoint_dir: Optional[str] = None,
    ) -> TrainingResult:
        optimizer = self._build_optimizer()
        scheduler, scheduler_needs_loss = self._build_scheduler(optimizer)

        history: Dict[str, List[float]] = {"loss": []}
        best_loss = float("inf")
        best_epoch = -1
        best_state = None

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()

            outputs = self.graph.forward(
                external_inputs=external_inputs,
                n_timesteps=n_timesteps,
                dt=dt,
            )
            total_loss, diagnostics = self.loss.compute(outputs)

            if self.spatial_reg > 0:
                total_loss = total_loss + self.param_manager.spatial_regularization_loss(
                    self.spatial_reg
                )

            total_loss.backward()

            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.graph.get_all_parameters(),
                    self.grad_clip,
                )
            optimizer.step()

            if scheduler is not None:
                if scheduler_needs_loss:
                    scheduler.step(total_loss.detach())
                else:
                    scheduler.step()

            history["loss"].append(float(total_loss.detach().cpu().item()))
            for key, value in diagnostics.items():
                history.setdefault(key, []).append(value)

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_epoch = epoch
                best_state = self._clone_parameters(self._collect_parameters())

            if checkpoint_dir and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                self.param_manager.save_checkpoint(ckpt_path)

            if verbose and (epoch + 1) % max(1, min(10, self.n_epochs // 10)) == 0:
                logger.info(f"Epoch {epoch+1:3d} | Loss: {total_loss.item():.6f}")

        final_params = self._collect_parameters()
        result = TrainingResult(
            history=history,
            best_loss=best_loss,
            best_epoch=best_epoch,
            final_parameters=final_params,
            convergence_info={},
        )
        return result
