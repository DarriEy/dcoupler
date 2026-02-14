"""BMI-aligned lifecycle mixin for dCoupler components.

The Basic Model Interface (BMI) is a standardized interface for model
coupling.  This mixin maps BMI lifecycle methods to the DifferentiableComponent
protocol so that existing components can be used in BMI-based workflows
without modifying their core implementation.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List


class BMIMixin(abc.ABC):
    """BMI-aligned lifecycle for dCoupler components.

    Subclasses implement these methods to expose a BMI-style interface
    while still participating in the dCoupler CouplingGraph.

    BMI ↔ DifferentiableComponent mapping:
        bmi_initialize(config)    → initialize(config) + get_initial_state()
        bmi_update(inputs, dt)    → step(inputs, state, dt)
        bmi_update_batch(…, n)    → run(inputs, state, dt, n)
        bmi_get_state()           → return internal state tensor
        bmi_set_state(state)      → set internal state tensor
        bmi_get_value(name)       → index into last output dict
        bmi_finalize()            → cleanup hook
    """

    @abc.abstractmethod
    def bmi_initialize(self, config: dict) -> None:
        """One-time setup: parse config, allocate state, load data."""

    @abc.abstractmethod
    def bmi_update(self, inputs: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """Advance one timestep, return outputs dict."""

    def bmi_update_batch(
        self, inputs: Dict[str, Any], dt: float, n_timesteps: int
    ) -> Dict[str, Any]:
        """Advance multiple timesteps. Default: loop bmi_update."""
        all_outputs: Dict[str, list] = {}
        for _ in range(n_timesteps):
            outputs = self.bmi_update(inputs, dt)
            for k, v in outputs.items():
                all_outputs.setdefault(k, []).append(v)
        return all_outputs

    @abc.abstractmethod
    def bmi_finalize(self) -> None:
        """Release resources."""

    @abc.abstractmethod
    def bmi_get_state(self) -> Any:
        """Return the current internal state."""

    @abc.abstractmethod
    def bmi_set_state(self, state: Any) -> None:
        """Overwrite the current internal state."""

    @abc.abstractmethod
    def bmi_get_value(self, name: str) -> Any:
        """Get a named output/state variable by name."""

    @abc.abstractmethod
    def bmi_set_value(self, name: str, value: Any) -> None:
        """Set a named input variable by name."""

    @abc.abstractmethod
    def bmi_get_output_var_names(self) -> List[str]:
        """Return list of output variable names."""

    @abc.abstractmethod
    def bmi_get_input_var_names(self) -> List[str]:
        """Return list of input variable names."""
