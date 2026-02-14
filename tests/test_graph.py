"""Tests for CouplingGraph core functionality."""

import pytest
import torch
import torch.nn as nn

from dcoupler.core.component import (
    DifferentiableComponent,
    FluxDirection,
    FluxSpec,
    GradientMethod,
    ParameterSpec,
)
from dcoupler.core.graph import CouplingGraph


class SimpleBucket(DifferentiableComponent):
    def __init__(self, name="bucket"):
        self._name = name
        self._k = nn.Parameter(torch.tensor(0.0))

    @property
    def name(self):
        return self._name

    @property
    def input_fluxes(self):
        return [FluxSpec("precip", "mm/d", FluxDirection.INPUT, "hru", 86400, ("time",))]

    @property
    def output_fluxes(self):
        return [
            FluxSpec("runoff", "mm/d", FluxDirection.OUTPUT, "hru", 86400, ("time",),
                     conserved_quantity="water_mass"),
        ]

    @property
    def parameters(self):
        return [ParameterSpec("k", 0.01, 0.99)]

    @property
    def gradient_method(self):
        return GradientMethod.AUTOGRAD

    @property
    def state_size(self):
        return 1

    def get_initial_state(self):
        return torch.tensor([10.0])

    def step(self, inputs, state, dt):
        k = torch.sigmoid(self._k)
        S = state[0]
        runoff = k * S
        S_new = torch.clamp(S + inputs["precip"] - runoff, min=0.0)
        return {"runoff": runoff}, torch.stack([S_new])

    def get_torch_parameters(self):
        return [self._k]

    def get_physical_parameters(self):
        return {"k": torch.sigmoid(self._k)}


class SimpleRouter(DifferentiableComponent):
    def __init__(self, name="router"):
        self._name = name
        self._alpha = nn.Parameter(torch.tensor(0.0))

    @property
    def name(self):
        return self._name

    @property
    def input_fluxes(self):
        return [FluxSpec("lateral_inflow", "mm/d", FluxDirection.INPUT, "hru", 86400, ("time",))]

    @property
    def output_fluxes(self):
        return [FluxSpec("discharge", "mm/d", FluxDirection.OUTPUT, "point", 86400, ("time",))]

    @property
    def parameters(self):
        return [ParameterSpec("alpha", 0.01, 0.99)]

    @property
    def gradient_method(self):
        return GradientMethod.AUTOGRAD

    @property
    def state_size(self):
        return 1

    def get_initial_state(self):
        return torch.tensor([0.0])

    def step(self, inputs, state, dt):
        alpha = torch.sigmoid(self._alpha)
        Q_in = inputs["lateral_inflow"]
        Q_prev = state[0]
        Q_out = (1 - alpha) * Q_in + alpha * Q_prev
        return {"discharge": Q_out}, torch.stack([Q_out])

    def get_torch_parameters(self):
        return [self._alpha]

    def get_physical_parameters(self):
        return {"alpha": torch.sigmoid(self._alpha)}


class TestCouplingGraph:
    def test_add_component(self):
        graph = CouplingGraph()
        graph.add_component(SimpleBucket())
        assert "bucket" in graph.components

    def test_duplicate_component_raises(self):
        graph = CouplingGraph()
        graph.add_component(SimpleBucket())
        with pytest.raises(ValueError, match="already registered"):
            graph.add_component(SimpleBucket())

    def test_connect_components(self):
        graph = CouplingGraph()
        graph.add_component(SimpleBucket())
        graph.add_component(SimpleRouter())
        graph.connect("bucket", "runoff", "router", "lateral_inflow")
        assert len(graph.connections) == 1

    def test_connect_unknown_source_raises(self):
        graph = CouplingGraph()
        graph.add_component(SimpleBucket())
        with pytest.raises(ValueError, match="Unknown source"):
            graph.connect("nonexistent", "runoff", "bucket", "precip")

    def test_cycle_detection(self):
        graph = CouplingGraph()
        b1 = SimpleBucket("b1")
        b2 = SimpleBucket("b2")
        graph.add_component(b1)
        graph.add_component(b2)
        # b1 → b2 is fine
        graph.connect("b1", "runoff", "b2", "precip")
        # b2 → b1 would create a cycle
        with pytest.raises(ValueError, match="cycle"):
            graph.connect("b2", "runoff", "b1", "precip")

    def test_forward_stepwise(self):
        graph = CouplingGraph()
        graph.add_component(SimpleBucket())
        graph.add_component(SimpleRouter())
        graph.connect("bucket", "runoff", "router", "lateral_inflow")

        precip = torch.ones(10) * 5.0
        outputs = graph.forward(
            external_inputs={"bucket": {"precip": precip}},
            n_timesteps=10,
            dt=86400,
        )
        assert "bucket" in outputs
        assert "router" in outputs
        assert "runoff" in outputs["bucket"]
        assert "discharge" in outputs["router"]
        assert outputs["bucket"]["runoff"].shape[0] == 10
        assert outputs["router"]["discharge"].shape[0] == 10

    def test_gradient_flow(self):
        graph = CouplingGraph()
        bucket = SimpleBucket()
        router = SimpleRouter()
        graph.add_component(bucket)
        graph.add_component(router)
        graph.connect("bucket", "runoff", "router", "lateral_inflow")

        precip = torch.ones(5) * 5.0
        outputs = graph.forward(
            external_inputs={"bucket": {"precip": precip}},
            n_timesteps=5,
            dt=86400,
        )
        loss = outputs["router"]["discharge"].sum()
        loss.backward()

        assert bucket._k.grad is not None
        assert router._alpha.grad is not None

    def test_validate_unconnected_input(self):
        graph = CouplingGraph()
        graph.add_component(SimpleBucket())
        graph.add_component(SimpleRouter())
        # Don't connect them
        warnings = graph.validate()
        assert any("Unconnected" in w for w in warnings)

    def test_get_all_parameters(self):
        graph = CouplingGraph()
        graph.add_component(SimpleBucket())
        graph.add_component(SimpleRouter())
        params = graph.get_all_parameters()
        assert len(params) == 2
