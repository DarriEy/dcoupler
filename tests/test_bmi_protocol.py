"""Tests for BMI protocol mixin."""

import pytest
import torch
import torch.nn as nn

from dcoupler.core.bmi import BMIMixin
from dcoupler.core.component import (
    DifferentiableComponent,
    FluxDirection,
    FluxSpec,
    GradientMethod,
    ParameterSpec,
)


class ToyBMIComponent(DifferentiableComponent, BMIMixin):
    """Minimal BMI-compliant component for testing."""

    def __init__(self, name="toy"):
        self._name = name
        self._state = None
        self._last_outputs = {}
        self._config = {}
        self._k = nn.Parameter(torch.tensor(0.5))

    @property
    def name(self):
        return self._name

    @property
    def input_fluxes(self):
        return [FluxSpec("precip", "mm/d", FluxDirection.INPUT, "hru", 86400, ("time",))]

    @property
    def output_fluxes(self):
        return [FluxSpec("runoff", "mm/d", FluxDirection.OUTPUT, "hru", 86400, ("time",))]

    @property
    def parameters(self):
        return [ParameterSpec("k", 0.0, 1.0)]

    @property
    def gradient_method(self):
        return GradientMethod.AUTOGRAD

    @property
    def state_size(self):
        return 1

    def initialize(self, config):
        self._config = config

    def get_initial_state(self):
        return torch.tensor([10.0])

    def step(self, inputs, state, dt):
        runoff = self._k * state[0]
        new_state = torch.stack([state[0] + inputs["precip"] - runoff])
        return {"runoff": runoff}, new_state

    def get_torch_parameters(self):
        return [self._k]

    def get_physical_parameters(self):
        return {"k": self._k}

    # BMI implementation
    def bmi_initialize(self, config):
        self.initialize(config)
        self._state = self.get_initial_state()

    def bmi_update(self, inputs, dt):
        tensor_inputs = {k: torch.tensor(v, dtype=torch.float32) for k, v in inputs.items()}
        outputs, self._state = self.step(tensor_inputs, self._state, dt)
        self._last_outputs = outputs
        return {k: float(v) for k, v in outputs.items()}

    def bmi_update_batch(self, inputs, dt, n_timesteps):
        all_out = {}
        for t in range(n_timesteps):
            t_inputs = {k: v[t] if hasattr(v, '__getitem__') else v for k, v in inputs.items()}
            out = self.bmi_update(t_inputs, dt)
            for k, v in out.items():
                all_out.setdefault(k, []).append(v)
        return all_out

    def bmi_finalize(self):
        self._state = None

    def bmi_get_state(self):
        return self._state

    def bmi_set_state(self, state):
        self._state = state

    def bmi_get_value(self, name):
        return self._last_outputs.get(name)

    def bmi_set_value(self, name, value):
        pass

    def bmi_get_output_var_names(self):
        return ["runoff"]

    def bmi_get_input_var_names(self):
        return ["precip"]


class TestBMIProtocol:
    def test_initialize(self):
        comp = ToyBMIComponent()
        comp.bmi_initialize({"test": True})
        assert comp._state is not None
        assert comp._config == {"test": True}

    def test_update_single_step(self):
        comp = ToyBMIComponent()
        comp.bmi_initialize({})
        result = comp.bmi_update({"precip": 5.0}, dt=86400)
        assert "runoff" in result
        assert isinstance(result["runoff"], float)
        assert result["runoff"] > 0

    def test_update_batch(self):
        comp = ToyBMIComponent()
        comp.bmi_initialize({})
        precip = [5.0, 3.0, 7.0]
        result = comp.bmi_update_batch({"precip": precip}, dt=86400, n_timesteps=3)
        assert "runoff" in result
        assert len(result["runoff"]) == 3

    def test_state_get_set(self):
        comp = ToyBMIComponent()
        comp.bmi_initialize({})
        original_state = comp.bmi_get_state().clone()
        new_state = torch.tensor([99.0])
        comp.bmi_set_state(new_state)
        assert torch.allclose(comp.bmi_get_state(), new_state)

    def test_finalize(self):
        comp = ToyBMIComponent()
        comp.bmi_initialize({})
        assert comp._state is not None
        comp.bmi_finalize()
        assert comp._state is None

    def test_var_names(self):
        comp = ToyBMIComponent()
        assert comp.bmi_get_output_var_names() == ["runoff"]
        assert comp.bmi_get_input_var_names() == ["precip"]

    def test_get_value_after_update(self):
        comp = ToyBMIComponent()
        comp.bmi_initialize({})
        comp.bmi_update({"precip": 5.0}, dt=86400)
        val = comp.bmi_get_value("runoff")
        assert val is not None

    def test_lifecycle_sequence(self):
        comp = ToyBMIComponent()
        comp.bmi_initialize({"param": 1})
        for t in range(5):
            comp.bmi_update({"precip": float(t)}, dt=86400)
        state = comp.bmi_get_state()
        assert state is not None
        comp.bmi_finalize()
        assert comp._state is None
