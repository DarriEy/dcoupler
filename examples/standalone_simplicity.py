from __future__ import annotations

import torch

import dcoupler as dc


class SimpleBucket(dc.DifferentiableComponent):
    def __init__(self, n_hrus: int, dt: float = 86400.0) -> None:
        self._name = "land"
        self.n_hrus = n_hrus
        self.dt = dt
        self._param = torch.nn.Parameter(torch.tensor(0.5))

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_fluxes(self):
        return [
            dc.FluxSpec(
                name="precip",
                units="mm/day",
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
                name="runoff",
                units="mm/day",
                direction=dc.FluxDirection.OUTPUT,
                spatial_type="hru",
                temporal_resolution=self.dt,
                dims=("time", "hru"),
                conserved_quantity="water_mass",
            )
        ]

    @property
    def parameters(self):
        return [
            dc.ParameterSpec(
                name="runoff_coef",
                lower_bound=0.0,
                upper_bound=1.0,
                spatial=False,
            )
        ]

    @property
    def gradient_method(self):
        return dc.GradientMethod.AUTOGRAD

    @property
    def state_size(self):
        return 0

    def get_initial_state(self) -> torch.Tensor:
        return torch.empty(0)

    def step(self, inputs, state, dt):
        precip = inputs["precip"]
        runoff = precip * torch.sigmoid(self._param)
        return {"runoff": runoff}, state

    def get_torch_parameters(self):
        return [self._param]

    def get_physical_parameters(self):
        return {"runoff_coef": torch.sigmoid(self._param)}


class LagRouter(dc.DifferentiableComponent):
    def __init__(self, n_reaches: int, dt: float = 86400.0) -> None:
        self._name = "routing"
        self.n_reaches = n_reaches
        self.dt = dt

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_fluxes(self):
        return [
            dc.FluxSpec(
                name="lateral_inflow",
                units="mm/day",
                direction=dc.FluxDirection.INPUT,
                spatial_type="reach",
                temporal_resolution=self.dt,
                dims=("time", "reach"),
            )
        ]

    @property
    def output_fluxes(self):
        return [
            dc.FluxSpec(
                name="discharge",
                units="mm/day",
                direction=dc.FluxDirection.OUTPUT,
                spatial_type="reach",
                temporal_resolution=self.dt,
                dims=("time", "reach"),
                conserved_quantity="water_mass",
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

    def get_initial_state(self) -> torch.Tensor:
        return torch.empty(0)

    def step(self, inputs, state, dt):
        return {"discharge": inputs["lateral_inflow"]}, state

    def get_torch_parameters(self):
        return []

    def get_physical_parameters(self):
        return {}


def main():
    torch.manual_seed(0)
    n = 10
    timesteps = 30

    land = SimpleBucket(n_hrus=n)
    router = LagRouter(n_reaches=n)

    graph = dc.CouplingGraph()
    graph.add_component(land)
    graph.add_component(router)
    graph.connect(
        "land",
        "runoff",
        "routing",
        "lateral_inflow",
        spatial_remap=dc.SpatialRemapper.identity(n),
    )

    forcing = torch.rand(timesteps, n)
    observed = forcing[:, -1] * 0.6

    obs_operator = dc.StreamflowObserver(gauge_reach_ids=[-1], component="routing")
    loss = dc.MultiObservationLoss()
    loss.add_term(obs_operator, observed, loss_fn="nse")

    trainer = dc.Trainer(graph, loss, n_epochs=10, lr=0.1)
    result = trainer.train(
        external_inputs={"land": {"precip": forcing}},
        n_timesteps=timesteps,
        dt=86400.0,
        verbose=True,
    )

    print("Best loss:", result.best_loss)


if __name__ == "__main__":
    main()
