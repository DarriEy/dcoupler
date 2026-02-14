"""ProcessComponent for wrapping external executables as dCoupler components.

External models (SUMMA, MESH, CLM, mizuRoute, ParFlow, MODFLOW) run as
subprocesses with file I/O marshalling.  They participate in the coupling
graph but do not provide native gradients (unless finite-difference is
enabled).
"""

from __future__ import annotations

import abc
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from dcoupler.core.component import (
    DifferentiableComponent,
    GradientMethod,
    ParameterSpec,
)
from dcoupler.core.bmi import BMIMixin


class ProcessComponent(DifferentiableComponent, BMIMixin):
    """Base class for external-executable model components.

    Subclasses implement ``write_inputs``, ``execute``, and ``read_outputs``
    to marshal data to/from the subprocess.  ``step()`` raises because
    process-based models require full-sequence (batch) execution.
    """

    def __init__(
        self,
        name: str,
        work_dir: Optional[Path] = None,
        gradient_method: GradientMethod = GradientMethod.NONE,
    ):
        self._name = name
        self._work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        self._gradient_method = gradient_method
        self._last_outputs: Dict[str, torch.Tensor] = {}
        self._state: Optional[torch.Tensor] = None
        self._config: dict = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def gradient_method(self) -> GradientMethod:
        return self._gradient_method

    @property
    def state_size(self) -> int:
        return 0

    @property
    def requires_batch(self) -> bool:
        return True

    @property
    def parameters(self) -> List[ParameterSpec]:
        return []

    def initialize(self, config: dict) -> None:
        self._config = config

    def get_initial_state(self) -> torch.Tensor:
        return torch.empty(0)

    @abc.abstractmethod
    def write_inputs(
        self, inputs: Dict[str, torch.Tensor], work_dir: Path
    ) -> None:
        """Write input tensors to files that the external model reads."""

    @abc.abstractmethod
    def execute(self, work_dir: Path) -> int:
        """Run the external model. Return exit code (0 = success)."""

    @abc.abstractmethod
    def read_outputs(self, work_dir: Path) -> Dict[str, torch.Tensor]:
        """Read model outputs from files and return as tensors."""

    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        state: torch.Tensor,
        dt: float,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        raise RuntimeError(
            "ProcessComponent requires batch execution via run(). "
            "Set requires_batch=True and use CouplingGraph._forward_batch()."
        )

    def run(
        self,
        inputs: Dict[str, torch.Tensor],
        state: torch.Tensor,
        dt: float,
        n_timesteps: int,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if self._gradient_method == GradientMethod.FINITE_DIFFERENCE:
            return FiniteDifferenceProcess.apply_process(
                self, inputs, state, dt, n_timesteps
            )
        self.write_inputs(inputs, self._work_dir)
        exit_code = self.execute(self._work_dir)
        if exit_code != 0:
            raise RuntimeError(
                f"ProcessComponent '{self._name}' failed with exit code {exit_code}"
            )
        outputs = self.read_outputs(self._work_dir)
        self._last_outputs = outputs
        return outputs, state

    def get_torch_parameters(self) -> List[nn.Parameter]:
        return []

    def get_physical_parameters(self) -> Dict[str, torch.Tensor]:
        return {}

    # -- BMI interface implementation --------------------------------------

    def bmi_initialize(self, config: dict) -> None:
        self.initialize(config)
        self._state = self.get_initial_state()

    def bmi_update(self, inputs: Dict[str, Any], dt: float) -> Dict[str, Any]:
        raise RuntimeError("ProcessComponent requires batch execution via bmi_update_batch()")

    def bmi_update_batch(
        self, inputs: Dict[str, Any], dt: float, n_timesteps: int
    ) -> Dict[str, Any]:
        tensor_inputs = {
            k: torch.as_tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        state = self._state if self._state is not None else self.get_initial_state()
        outputs, self._state = self.run(tensor_inputs, state, dt, n_timesteps)
        self._last_outputs = outputs
        return {k: v.detach().numpy() for k, v in outputs.items()}

    def bmi_finalize(self) -> None:
        pass

    def bmi_get_state(self) -> Any:
        return self._state

    def bmi_set_state(self, state: Any) -> None:
        self._state = state

    def bmi_get_value(self, name: str) -> Any:
        if name in self._last_outputs:
            return self._last_outputs[name].detach().numpy()
        raise KeyError(f"Unknown output variable '{name}'")

    def bmi_set_value(self, name: str, value: Any) -> None:
        pass  # Process components receive inputs via write_inputs

    def bmi_get_output_var_names(self) -> List[str]:
        return [f.name for f in self.output_fluxes]

    def bmi_get_input_var_names(self) -> List[str]:
        return [f.name for f in self.input_fluxes]


class FiniteDifferenceProcess:
    """Wraps ProcessComponent.run() with finite-difference gradient estimation.

    This is not a torch.autograd.Function because process components
    typically don't have torch.nn.Parameters.  Instead, this provides
    a utility for computing parameter sensitivities externally.
    """

    @staticmethod
    def apply_process(
        component: ProcessComponent,
        inputs: Dict[str, torch.Tensor],
        state: torch.Tensor,
        dt: float,
        n_timesteps: int,
        epsilon: float = 1e-4,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Run the process component (forward only, no autograd)."""
        component.write_inputs(inputs, component._work_dir)
        exit_code = component.execute(component._work_dir)
        if exit_code != 0:
            raise RuntimeError(
                f"ProcessComponent '{component._name}' failed with exit code {exit_code}"
            )
        outputs = component.read_outputs(component._work_dir)
        return outputs, state

    @staticmethod
    def estimate_gradient(
        component: ProcessComponent,
        inputs: Dict[str, torch.Tensor],
        state: torch.Tensor,
        dt: float,
        n_timesteps: int,
        param_name: str,
        param_values: np.ndarray,
        epsilon: float = 0.01,
    ) -> np.ndarray:
        """Estimate dOutput/dParam via central finite differences.

        Args:
            component: The process component to perturb
            inputs: Input tensors for the run
            state: Initial state
            dt: Timestep
            n_timesteps: Number of timesteps
            param_name: Name of parameter to perturb
            param_values: Current parameter values (1D array)
            epsilon: Relative perturbation size

        Returns:
            Gradient array of shape [n_outputs, n_params]
        """
        gradients = []
        for i in range(len(param_values)):
            perturbed_plus = param_values.copy()
            perturbed_plus[i] *= (1 + epsilon)
            perturbed_minus = param_values.copy()
            perturbed_minus[i] *= (1 - epsilon)

            component.bmi_set_value(param_name, perturbed_plus)
            outputs_plus, _ = component.run(inputs, state, dt, n_timesteps)

            component.bmi_set_value(param_name, perturbed_minus)
            outputs_minus, _ = component.run(inputs, state, dt, n_timesteps)

            component.bmi_set_value(param_name, param_values)

            grad_i = {}
            for key in outputs_plus:
                diff = outputs_plus[key] - outputs_minus[key]
                denom = 2 * epsilon * param_values[i]
                grad_i[key] = (diff / denom).detach().numpy()
            gradients.append(grad_i)

        return gradients
