from __future__ import annotations

import torch

import dcoupler as dc


class Producer(dc.DifferentiableComponent):
    def __init__(self, name: str, n: int, dt: float) -> None:
        self._name = name
        self.n = n
        self.dt = dt

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
                temporal_resolution=self.dt,
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
        return {"flux": torch.ones(self.n)}, state

    def get_torch_parameters(self):
        return []

    def get_physical_parameters(self):
        return {}


class Consumer(dc.DifferentiableComponent):
    def __init__(self, name: str, n: int, dt: float) -> None:
        self._name = name
        self.n = n
        self.dt = dt

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
                temporal_resolution=self.dt,
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
                temporal_resolution=self.dt,
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
        return torch.zeros(self.n)

    def step(self, inputs, state, dt):
        state = state + inputs["flux"]
        return {"accum": state}, state

    def get_torch_parameters(self):
        return []

    def get_physical_parameters(self):
        return {}


if __name__ == "__main__":
    outer_dt = 4.0
    producer_dt = 4.0
    consumer_dt = 1.0

    prod = Producer("prod", n=5, dt=producer_dt)
    cons = Consumer("cons", n=5, dt=consumer_dt)

    graph = dc.CouplingGraph()
    graph.add_component(prod)
    graph.add_component(cons)
    graph.connect("prod", "flux", "cons", "flux", temporal_interp="step")

    temporal = dc.TemporalOrchestrator(outer_dt=outer_dt)
    temporal.set_component_dt("prod", producer_dt)
    temporal.set_component_dt("cons", consumer_dt)

    outputs = graph.forward(external_inputs={}, n_timesteps=3, dt=outer_dt, temporal=temporal)
    print(outputs["cons"]["accum"].shape)
    print(outputs["cons"]["accum"])
