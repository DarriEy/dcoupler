# dCoupler

Differentiable Earth System Coupler — a PyTorch-based framework for building
differentiable coupling graphs between hydrological and earth system model
components.

## Quickstart

```bash
pip install dcoupler
```

```python
import torch
import dcoupler as dc

# Define two components
class Bucket(dc.DifferentiableComponent):
    ...

class Router(dc.DifferentiableComponent):
    ...

# Build coupling graph
graph = dc.CouplingGraph()
graph.add_component(bucket)
graph.add_component(router)
graph.connect("bucket", "runoff", "router", "lateral_inflow")

# Run forward
outputs = graph.forward(external_inputs, n_timesteps=365, dt=86400)

# Optimize via backprop
loss = loss_fn(outputs, observations)
loss.backward()
optimizer.step()
```

## Component Types

| Type | Gradient | Use Case |
|------|----------|----------|
| `DifferentiableComponent` | PyTorch autograd | Pure PyTorch models |
| `JAXComponent` | JAX vjp via bridge | JAX models (XAJ, SAC-SMA, Snow-17) |
| `ProcessComponent` | None / finite diff | External executables (SUMMA, MESH) |
| `EnzymeComponent` | Enzyme AD | C++ models with Enzyme |

All component types implement the `BMIMixin` protocol for standardized
lifecycle management (initialize/update/finalize).

## Optional Dependencies

```bash
pip install dcoupler[jax]     # JAX bridge support
pip install dcoupler[models]  # cfuse + droute components
pip install dcoupler[all]     # Everything
```

## License

MIT
