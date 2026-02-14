from __future__ import annotations

import torch

import dcoupler as dc


def test_temporal_orchestrator_runs():
    class Producer(dc.DifferentiableComponent):
        def __init__(self):
            self._name = "prod"

        @property
        def name(self):
            return self._name

        @property
        def input_fluxes(self):
            return []

        @property
        def output_fluxes(self):
            return [
                dc.FluxSpec(
                    name="flux",
                    units="unit",
                    direction=dc.FluxDirection.OUTPUT,
                    spatial_type="hru",
                    temporal_resolution=4.0,
                    dims=("time", "hru"),
                )
            ]

        @property
        def parameters(self):
            return []

        @property
        def gradient_method(self):
            return dc.GradientMethod.AUTOGRAD

        @property
        def state_size(self):
            return 0

        def get_initial_state(self):
            return torch.empty(0)

        def step(self, inputs, state, dt):
            return {"flux": torch.ones(2)}, state

        def get_torch_parameters(self):
            return []

        def get_physical_parameters(self):
            return {}

    class Consumer(dc.DifferentiableComponent):
        def __init__(self):
            self._name = "cons"

        @property
        def name(self):
            return self._name

        @property
        def input_fluxes(self):
            return [
                dc.FluxSpec(
                    name="flux",
                    units="unit",
                    direction=dc.FluxDirection.INPUT,
                    spatial_type="hru",
                    temporal_resolution=1.0,
                    dims=("time", "hru"),
                )
            ]

        @property
        def output_fluxes(self):
            return [
                dc.FluxSpec(
                    name="accum",
                    units="unit",
                    direction=dc.FluxDirection.OUTPUT,
                    spatial_type="hru",
                    temporal_resolution=1.0,
                    dims=("time", "hru"),
                )
            ]

        @property
        def parameters(self):
            return []

        @property
        def gradient_method(self):
            return dc.GradientMethod.AUTOGRAD

        @property
        def state_size(self):
            return 0

        def get_initial_state(self):
            return torch.zeros(2)

        def step(self, inputs, state, dt):
            state = state + inputs["flux"]
            return {"accum": state}, state

        def get_torch_parameters(self):
            return []

        def get_physical_parameters(self):
            return {}

    graph = dc.CouplingGraph()
    graph.add_component(Producer())
    graph.add_component(Consumer())
    graph.connect("prod", "flux", "cons", "flux", temporal_interp="step")

    temporal = dc.TemporalOrchestrator(outer_dt=4.0)
    temporal.set_component_dt("prod", 4.0)
    temporal.set_component_dt("cons", 1.0)

    outputs = graph.forward(external_inputs={}, n_timesteps=2, dt=4.0, temporal=temporal)
    accum = outputs["cons"]["accum"]
    assert accum.shape[0] == 2
    assert torch.all(accum[-1] > 0)
