from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import abc
import torch


class FluxDirection(Enum):
    INPUT = "input"
    OUTPUT = "output"
    BIDIRECTIONAL = "bidirectional"


@dataclass(frozen=True)
class FluxSpec:
    """Declaration of a single flux variable at a component boundary."""

    name: str
    units: str
    direction: FluxDirection
    spatial_type: str
    temporal_resolution: float
    dims: Tuple[str, ...]
    optional: bool = False
    conserved_quantity: Optional[str] = None


@dataclass(frozen=True)
class ParameterSpec:
    """Declaration of an optimizable parameter."""

    name: str
    lower_bound: float
    upper_bound: float
    spatial: bool = False
    n_spatial: Optional[int] = None
    log_transform: bool = False


class GradientMethod(Enum):
    AUTOGRAD = "autograd"
    ENZYME = "enzyme"
    ADJOINT = "adjoint"
    FINITE_DIFFERENCE = "finite_diff"
    NONE = "none"


class DifferentiableComponent(abc.ABC):
    """
    Protocol for a differentiable model component.

    Implementations wrap specific models and expose a uniform interface
    for the coupling graph.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique component identifier."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def input_fluxes(self) -> List[FluxSpec]:
        """Fluxes this component requires as input."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_fluxes(self) -> List[FluxSpec]:
        """Fluxes this component produces as output."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def parameters(self) -> List[ParameterSpec]:
        """Optimizable parameters exposed by this component."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def gradient_method(self) -> GradientMethod:
        """How this component provides gradients."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def state_size(self) -> int:
        """Number of internal state variables."""
        raise NotImplementedError

    @property
    def requires_batch(self) -> bool:
        """
        Whether this component must be executed over a full time sequence.
        Components backed by batch-only Enzyme/JAX wrappers should set this.
        """
        return False

    def initialize(self, config: dict) -> None:
        """One-time setup (optional)."""
        return None

    @abc.abstractmethod
    def get_initial_state(self) -> torch.Tensor:
        """Return initial state tensor [n_spatial, n_states]."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        state: torch.Tensor,
        dt: float,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Advance component by one timestep.

        Returns:
            outputs: Dict mapping flux name -> tensor
            new_state: Updated state tensor [n_spatial, n_states]
        """
        raise NotImplementedError

    def run(
        self,
        inputs: Dict[str, torch.Tensor],
        state: torch.Tensor,
        dt: float,
        n_timesteps: int,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Optional full-sequence execution. Default falls back to step-by-step.
        """
        outputs = {spec.name: [] for spec in self.output_fluxes}
        current_state = state
        for t in range(n_timesteps):
            step_inputs = {
                k: (v[t] if v.dim() > 1 else v)
                for k, v in inputs.items()
            }
            step_out, current_state = self.step(step_inputs, current_state, dt)
            for name, tensor in step_out.items():
                outputs[name].append(tensor)
        stacked = {k: torch.stack(v, dim=0) for k, v in outputs.items()}
        return stacked, current_state

    @abc.abstractmethod
    def get_torch_parameters(self) -> List[torch.nn.Parameter]:
        """Return list of optimizable PyTorch parameters."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_physical_parameters(self) -> Dict[str, torch.Tensor]:
        """Return current physical parameter values (after transforms)."""
        raise NotImplementedError
