"""Tests for ConservationChecker integration in CouplingGraph."""

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
from dcoupler.core.conservation import ConservationChecker
from dcoupler.core.connection import FluxConnection


class SourceComponent(DifferentiableComponent):
    """Produces a known output flux."""

    def __init__(self, name, output_value=1.0):
        self._name = name
        self._output_value = output_value
        self._k = nn.Parameter(torch.tensor(1.0))

    @property
    def name(self):
        return self._name

    @property
    def input_fluxes(self):
        return [FluxSpec("input", "mm/d", FluxDirection.INPUT, "hru", 86400, ("time",))]

    @property
    def output_fluxes(self):
        return [
            FluxSpec("flux_out", "mm/d", FluxDirection.OUTPUT, "hru", 86400, ("time",),
                     conserved_quantity="water_mass"),
        ]

    @property
    def parameters(self):
        return [ParameterSpec("k", 0.1, 10.0)]

    @property
    def gradient_method(self):
        return GradientMethod.AUTOGRAD

    @property
    def state_size(self):
        return 0

    def get_initial_state(self):
        return torch.empty(0)

    def step(self, inputs, state, dt):
        return {"flux_out": inputs["input"] * self._k}, state

    def get_torch_parameters(self):
        return [self._k]

    def get_physical_parameters(self):
        return {"k": self._k}


class SinkComponent(DifferentiableComponent):
    """Receives a flux and passes it through."""

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def input_fluxes(self):
        return [FluxSpec("flux_in", "mm/d", FluxDirection.INPUT, "hru", 86400, ("time",))]

    @property
    def output_fluxes(self):
        return [FluxSpec("output", "mm/d", FluxDirection.OUTPUT, "hru", 86400, ("time",))]

    @property
    def parameters(self):
        return []

    @property
    def gradient_method(self):
        return GradientMethod.NONE

    @property
    def state_size(self):
        return 0

    def get_initial_state(self):
        return torch.empty(0)

    def step(self, inputs, state, dt):
        return {"output": inputs["flux_in"]}, state

    def get_torch_parameters(self):
        return []

    def get_physical_parameters(self):
        return {}


class TestConservationChecker:
    def test_check_mode_logs_error(self):
        checker = ConservationChecker(mode="check")
        conn = FluxConnection(
            source_component="src", source_flux="out",
            target_component="tgt", target_flux="in",
            conserved_quantity="water_mass",
        )
        source = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, 2.2, 3.3])  # ~10% error
        areas = torch.ones(3)

        result = checker.check_connection(conn, source, target, areas, areas, 1.0)
        assert result is None  # check mode doesn't correct
        assert len(checker.conservation_log) == 1
        assert checker.conservation_log[0]["relative_error"] > 0

    def test_enforce_mode_corrects(self):
        checker = ConservationChecker(mode="enforce")
        conn = FluxConnection(
            source_component="src", source_flux="out",
            target_component="tgt", target_flux="in",
            conserved_quantity="water_mass",
        )
        source = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 4.0, 6.0])  # 2x too much
        areas = torch.ones(3)

        corrected = checker.check_connection(conn, source, target, areas, areas, 1.0)
        assert corrected is not None
        # After correction, total should match source total
        assert abs(corrected.sum().item() - source.sum().item()) < 1e-4

    def test_enforce_within_tolerance_no_correction(self):
        checker = ConservationChecker(mode="enforce", tolerance=0.01)
        conn = FluxConnection(
            source_component="src", source_flux="out",
            target_component="tgt", target_flux="in",
            conserved_quantity="water_mass",
        )
        source = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])  # exact match
        areas = torch.ones(3)

        corrected = checker.check_connection(conn, source, target, areas, areas, 1.0)
        # Within tolerance, returns the target as-is
        assert torch.allclose(corrected, target)


class TestConservationInGraph:
    def test_graph_with_conservation_check_mode(self):
        """Conservation checker in check mode should log but not modify outputs."""
        graph = CouplingGraph(conservation_mode="check")
        src = SourceComponent("source")
        sink = SinkComponent("sink")
        graph.add_component(src)
        graph.add_component(sink)
        graph.connect("source", "flux_out", "sink", "flux_in")

        inputs = {
            "source": {"input": torch.tensor([1.0, 2.0, 3.0])},
        }
        outputs = graph.forward(inputs, n_timesteps=3, dt=86400)
        assert "sink" in outputs
        assert "output" in outputs["sink"]
        # Check that conservation log was populated
        assert len(graph._conservation.conservation_log) > 0

    def test_graph_with_conservation_enforce_mode(self):
        """Conservation checker in enforce mode should correct outputs."""
        graph = CouplingGraph(conservation_mode="enforce")
        src = SourceComponent("source", output_value=1.0)
        sink = SinkComponent("sink")
        graph.add_component(src)
        graph.add_component(sink)
        graph.connect("source", "flux_out", "sink", "flux_in")

        inputs = {
            "source": {"input": torch.tensor([1.0, 2.0, 3.0])},
        }
        outputs = graph.forward(inputs, n_timesteps=3, dt=86400)
        assert "sink" in outputs

    def test_graph_without_conservation(self):
        """Without conservation mode, no checker should be active."""
        graph = CouplingGraph()
        assert graph._conservation is None

        src = SourceComponent("source")
        sink = SinkComponent("sink")
        graph.add_component(src)
        graph.add_component(sink)
        graph.connect("source", "flux_out", "sink", "flux_in")

        inputs = {
            "source": {"input": torch.tensor([1.0, 2.0, 3.0])},
        }
        outputs = graph.forward(inputs, n_timesteps=3, dt=86400)
        assert "sink" in outputs
