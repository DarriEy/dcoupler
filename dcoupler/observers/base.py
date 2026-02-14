from __future__ import annotations

import abc
from typing import Dict, List, Tuple

import torch


class ObservationOperator(abc.ABC):
    """Differentiable mapping from model space to observation space."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def required_model_outputs(self) -> List[Tuple[str, str]]:
        """List of (component_name, flux_name) required by this operator."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, model_outputs: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Return simulated observation from model outputs."""
        raise NotImplementedError

    def __call__(self, model_outputs: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        return self.apply(model_outputs)
