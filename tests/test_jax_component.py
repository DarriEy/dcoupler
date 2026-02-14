"""Tests for JAXComponent and JAX↔PyTorch gradient bridge.

These tests are skipped if JAX is not installed.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

jax_available = True
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax_available = False

from dcoupler.core.component import (
    FluxDirection,
    FluxSpec,
    GradientMethod,
    ParameterSpec,
)


pytestmark = pytest.mark.skipif(not jax_available, reason="JAX not installed")


@pytest.fixture
def jax_bridge():
    from dcoupler.wrappers.jax import JAXBridge
    return JAXBridge


@pytest.fixture
def jax_utils():
    from dcoupler.wrappers.jax import torch_to_jax, jax_to_torch
    return torch_to_jax, jax_to_torch


class TestTorchJAXConversion:
    def test_torch_to_jax_roundtrip(self, jax_utils):
        torch_to_jax, jax_to_torch = jax_utils
        t = torch.tensor([1.0, 2.0, 3.0])
        j = torch_to_jax(t)
        t2 = jax_to_torch(j)
        assert torch.allclose(t, t2)

    def test_2d_conversion(self, jax_utils):
        torch_to_jax, jax_to_torch = jax_utils
        t = torch.randn(3, 4)
        j = torch_to_jax(t)
        t2 = jax_to_torch(j)
        assert torch.allclose(t, t2, atol=1e-6)


class TestJAXBridge:
    def test_forward_simple_function(self, jax_bridge):
        """Test forward pass through a simple JAX function."""
        def jax_fn(x):
            return x ** 2

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        result = jax_bridge.apply(jax_fn, 1, x)
        expected = torch.tensor([4.0, 9.0])
        assert torch.allclose(result, expected)

    def test_backward_simple_gradient(self, jax_bridge):
        """Test that gradients flow back through the bridge."""
        def jax_fn(x):
            return jnp.sum(x ** 2)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        result = jax_bridge.apply(jax_fn, 1, x)
        result.backward()

        # d/dx(sum(x^2)) = 2x
        expected_grad = torch.tensor([4.0, 6.0])
        assert torch.allclose(x.grad, expected_grad)

    def test_gradient_two_args(self, jax_bridge):
        """Test gradients with two differentiable arguments."""
        def jax_fn(a, b):
            return jnp.sum(a * b)

        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0], requires_grad=True)
        result = jax_bridge.apply(jax_fn, 2, a, b)
        result.backward()

        # d/da(sum(a*b)) = b, d/db(sum(a*b)) = a
        assert torch.allclose(a.grad, torch.tensor([3.0, 4.0]))
        assert torch.allclose(b.grad, torch.tensor([1.0, 2.0]))


class TestJAXComponent:
    def test_create_component(self):
        from dcoupler.wrappers.jax import JAXComponent

        def jax_step(inputs, state, params, dt):
            k = params["k"]
            runoff = k * state
            new_state = state + inputs["precip"] - runoff
            return runoff, new_state

        comp = JAXComponent(
            name="test",
            jax_step_fn=jax_step,
            param_specs=[ParameterSpec("k", 0.01, 0.99)],
            state_size=1,
            input_flux_specs=[
                FluxSpec("precip", "mm/d", FluxDirection.INPUT, "hru", 86400, ("time",)),
            ],
            output_flux_specs=[
                FluxSpec("runoff", "mm/d", FluxDirection.OUTPUT, "hru", 86400, ("time",)),
            ],
        )

        assert comp.name == "test"
        assert comp.gradient_method == GradientMethod.AUTOGRAD
        assert len(comp.parameters) == 1
        assert len(comp.get_torch_parameters()) == 1

    def test_bmi_lifecycle(self):
        from dcoupler.wrappers.jax import JAXComponent

        def jax_step(inputs, state, params, dt):
            k = params["k"]
            return k * inputs["x"], state

        comp = JAXComponent(
            name="bmi_test",
            jax_step_fn=jax_step,
            param_specs=[ParameterSpec("k", 0.0, 1.0)],
            state_size=1,
            input_flux_specs=[
                FluxSpec("x", "m", FluxDirection.INPUT, "hru", 1, ("time",)),
            ],
            output_flux_specs=[
                FluxSpec("y", "m", FluxDirection.OUTPUT, "hru", 1, ("time",)),
            ],
        )

        comp.bmi_initialize({})
        assert comp.bmi_get_state() is not None
        assert comp.bmi_get_output_var_names() == ["y"]
        assert comp.bmi_get_input_var_names() == ["x"]
        comp.bmi_finalize()

    def test_physical_parameters(self):
        from dcoupler.wrappers.jax import JAXComponent

        def dummy(inputs, state, params, dt):
            return params["k"] * inputs["x"], state

        comp = JAXComponent(
            name="param_test",
            jax_step_fn=dummy,
            param_specs=[
                ParameterSpec("k", 0.0, 10.0),
            ],
            state_size=0,
            input_flux_specs=[
                FluxSpec("x", "m", FluxDirection.INPUT, "hru", 1, ("time",)),
            ],
            output_flux_specs=[
                FluxSpec("y", "m", FluxDirection.OUTPUT, "hru", 1, ("time",)),
            ],
        )

        phys = comp.get_physical_parameters()
        assert "k" in phys
        # Sigmoid(0) = 0.5, so k should be 0 + (10-0) * 0.5 = 5.0
        assert abs(phys["k"].item() - 5.0) < 0.01
