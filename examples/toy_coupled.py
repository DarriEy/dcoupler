"""Standalone toy coupling example using pure PyTorch components.

Demonstrates the CouplingGraph API without requiring cfuse or droute.
Two components: SimpleBucket (rainfall-runoff) and LagRouter (delay routing).
"""

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
    """Minimal single-bucket rainfall-runoff model.

    State: storage S (mm)
    Parameter: k (recession coefficient, 0-1)
    Physics: S_new = S + P - E - k*S; runoff = k*S
    """

    def __init__(self, name: str = "bucket"):
        self._name = name
        self._raw_k = nn.Parameter(torch.tensor(0.0))  # sigmoid → k ∈ (0, 1)

    @property
    def name(self):
        return self._name

    @property
    def input_fluxes(self):
        return [
            FluxSpec("precip", "mm/d", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("pet", "mm/d", FluxDirection.INPUT, "hru", 86400, ("time",), optional=True),
        ]

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
        return torch.tensor([10.0])  # initial storage = 10 mm

    def get_physical_parameters(self):
        return {"k": torch.sigmoid(self._raw_k)}

    def get_torch_parameters(self):
        return [self._raw_k]

    def step(self, inputs, state, dt):
        k = torch.sigmoid(self._raw_k)
        P = inputs["precip"]
        E = inputs.get("pet", torch.tensor(0.0))
        S = state[0]
        runoff = k * S
        S_new = torch.clamp(S + P - E - runoff, min=0.0)
        return {"runoff": runoff}, torch.stack([S_new])


class LagRouter(DifferentiableComponent):
    """Simple linear reservoir router: Q_out = (1 - alpha) * Q_in + alpha * Q_prev."""

    def __init__(self, name: str = "router"):
        self._name = name
        self._raw_alpha = nn.Parameter(torch.tensor(0.0))

    @property
    def name(self):
        return self._name

    @property
    def input_fluxes(self):
        return [
            FluxSpec("lateral_inflow", "mm/d", FluxDirection.INPUT, "hru", 86400, ("time",)),
        ]

    @property
    def output_fluxes(self):
        return [
            FluxSpec("discharge", "mm/d", FluxDirection.OUTPUT, "point", 86400, ("time",)),
        ]

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

    def get_physical_parameters(self):
        return {"alpha": torch.sigmoid(self._raw_alpha)}

    def get_torch_parameters(self):
        return [self._raw_alpha]

    def step(self, inputs, state, dt):
        alpha = torch.sigmoid(self._raw_alpha)
        Q_in = inputs["lateral_inflow"]
        Q_prev = state[0]
        Q_out = (1 - alpha) * Q_in + alpha * Q_prev
        return {"discharge": Q_out}, torch.stack([Q_out])


def main():
    torch.manual_seed(42)
    n_timesteps = 100

    # Synthetic forcing
    precip = torch.abs(torch.randn(n_timesteps)) * 5  # mm/d
    pet = torch.ones(n_timesteps) * 2.0  # mm/d

    # Synthetic observations (target discharge)
    obs = torch.abs(torch.randn(n_timesteps)) * 3

    # Build coupling graph
    bucket = SimpleBucket("bucket")
    router = LagRouter("router")

    graph = CouplingGraph()
    graph.add_component(bucket)
    graph.add_component(router)
    graph.connect("bucket", "runoff", "router", "lateral_inflow")

    warnings = graph.validate()
    if warnings:
        print("Validation warnings:", warnings)

    # Optimizer
    all_params = graph.get_all_parameters()
    optimizer = torch.optim.Adam(all_params, lr=0.05)

    # Training loop
    for epoch in range(50):
        optimizer.zero_grad()

        outputs = graph.forward(
            external_inputs={
                "bucket": {"precip": precip, "pet": pet},
            },
            n_timesteps=n_timesteps,
            dt=86400.0,
        )

        sim_Q = outputs["router"]["discharge"]
        loss = torch.mean((sim_Q - obs) ** 2)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            k = torch.sigmoid(bucket._raw_k).item()
            alpha = torch.sigmoid(router._raw_alpha).item()
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | k={k:.3f} alpha={alpha:.3f}")

    print("\nFinal parameters:")
    for name, comp in graph.components.items():
        print(f"  {name}: {comp.get_physical_parameters()}")


if __name__ == "__main__":
    main()
