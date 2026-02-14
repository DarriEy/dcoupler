from __future__ import annotations

import torch

import dcoupler as dc


def test_multi_observation_and_trainer_smoke():
    class DummyComponent(dc.DifferentiableComponent):
        def __init__(self) -> None:
            self._name = "dummy"
            self._param = torch.nn.Parameter(torch.tensor(0.1))

        @property
        def name(self) -> str:
            return self._name

        @property
        def input_fluxes(self):
            return []

        @property
        def output_fluxes(self):
            return [
                dc.FluxSpec(
                    name="signal",
                    units="unit",
                    direction=dc.FluxDirection.OUTPUT,
                    spatial_type="point",
                    temporal_resolution=1.0,
                    dims=("time",),
                )
            ]

        @property
        def parameters(self):
            return [
                dc.ParameterSpec(
                    name="scale",
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
            t = torch.ones(())
            return {"signal": t * torch.sigmoid(self._param)}, state

        def get_torch_parameters(self):
            return [self._param]

        def get_physical_parameters(self):
            return {"scale": torch.sigmoid(self._param)}

    graph = dc.CouplingGraph()
    comp = DummyComponent()
    graph.add_component(comp)

    class DummyObserver(dc.ObservationOperator):
        @property
        def name(self) -> str:
            return "dummy_obs"

        @property
        def required_model_outputs(self):
            return [("dummy", "signal")]

        def apply(self, model_outputs):
            return model_outputs["dummy"]["signal"]

    observed = torch.ones(5)
    obs = DummyObserver()
    loss = dc.MultiObservationLoss()
    loss.add_term(obs, observed, loss_fn="mse")

    trainer = dc.Trainer(graph, loss, n_epochs=2, lr=0.1)
    result = trainer.train(external_inputs={}, n_timesteps=5, dt=1.0, verbose=False)

    assert result.best_epoch >= 0
