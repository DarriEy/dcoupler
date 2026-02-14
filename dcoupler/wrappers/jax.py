"""JAXComponent for wrapping JAX-native models as dCoupler components.

JAX models (XAJ, SAC-SMA, Snow-17) are wrapped with a PyTorch↔JAX bridge
so their gradients flow through the CouplingGraph's autograd tape.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from dcoupler.core.component import (
    DifferentiableComponent,
    FluxDirection,
    FluxSpec,
    GradientMethod,
    ParameterSpec,
)
from dcoupler.core.bmi import BMIMixin


def _import_jax():
    """Lazy import of JAX to keep it optional."""
    try:
        import jax
        import jax.numpy as jnp
        return jax, jnp
    except ImportError:
        raise ImportError(
            "JAX is required for JAXComponent. Install with: pip install jax jaxlib"
        )


def torch_to_jax(tensor: torch.Tensor):
    """Convert a PyTorch tensor to a JAX array (zero-copy when possible)."""
    _, jnp = _import_jax()
    return jnp.array(tensor.detach().cpu().numpy())


def jax_to_torch(array, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert a JAX array to a PyTorch tensor."""
    result = torch.from_numpy(np.asarray(array)).float()
    if device is not None:
        result = result.to(device)
    return result


class JAXBridge(torch.autograd.Function):
    """torch.autograd.Function that bridges PyTorch ↔ JAX for gradient flow.

    Forward: converts torch tensors → JAX arrays, calls jax_fn, converts back.
    Backward: uses jax.vjp to compute vector-Jacobian products, converts back.
    """

    @staticmethod
    def forward(ctx, jax_fn, n_regular_args, *torch_args):
        """Forward pass through JAX function.

        Args:
            ctx: autograd context
            jax_fn: JAX function to call
            n_regular_args: number of tensor args that need gradients
            *torch_args: PyTorch tensors (first n_regular_args get gradients)
        """
        jax, _ = _import_jax()

        device = torch_args[0].device if len(torch_args) > 0 else None
        jax_args = tuple(torch_to_jax(a) for a in torch_args)

        differentiable_args = jax_args[:n_regular_args]
        static_args = jax_args[n_regular_args:]

        def fn_for_vjp(*diff_args):
            all_args = diff_args + static_args
            return jax_fn(*all_args)

        jax_out, vjp_fn = jax.vjp(fn_for_vjp, *differentiable_args)

        ctx.vjp_fn = vjp_fn
        ctx.n_regular_args = n_regular_args
        ctx.device = device
        ctx.jax_out_structure = jax.tree.structure(jax_out)

        if isinstance(jax_out, tuple):
            return tuple(jax_to_torch(o, device) for o in jax_out)
        return jax_to_torch(jax_out, device)

    @staticmethod
    def backward(ctx, *grad_outputs):
        jax_grads_out = tuple(torch_to_jax(g) for g in grad_outputs)

        if len(jax_grads_out) == 1:
            jax_grads_out = jax_grads_out[0]

        jax_grads_in = ctx.vjp_fn(jax_grads_out)

        torch_grads = tuple(
            jax_to_torch(g, ctx.device) for g in jax_grads_in
        )

        # None for jax_fn, None for n_regular_args, grads for differentiable, None for static
        n_static = 0  # will be inferred from total args
        result = (None, None) + torch_grads
        # Pad with None for static args
        total_expected = 2 + ctx.n_regular_args + (len(grad_outputs) - ctx.n_regular_args if len(grad_outputs) > ctx.n_regular_args else 0)
        while len(result) < 2 + ctx.n_regular_args:
            result = result + (None,)
        return result


class JAXBatchBridge(torch.autograd.Function):
    """Optimized bridge for batch (lax.scan) JAX functions."""

    @staticmethod
    def forward(ctx, jax_run_fn, n_param_tensors, *torch_args):
        """Forward pass through a JAX batch function (e.g. lax.scan wrapper).

        Args:
            ctx: autograd context
            jax_run_fn: JAX function that processes full timeseries
            n_param_tensors: number of leading args that are parameters (need grads)
            *torch_args: param tensors + input tensors + state + dt + n_timesteps
        """
        jax, _ = _import_jax()

        device = torch_args[0].device if torch_args else None
        jax_args = tuple(torch_to_jax(a) for a in torch_args)

        param_args = jax_args[:n_param_tensors]
        other_args = jax_args[n_param_tensors:]

        def fn_for_vjp(*p_args):
            return jax_run_fn(*(p_args + other_args))

        jax_out, vjp_fn = jax.vjp(fn_for_vjp, *param_args)

        ctx.vjp_fn = vjp_fn
        ctx.n_param_tensors = n_param_tensors
        ctx.device = device

        if isinstance(jax_out, tuple):
            return tuple(jax_to_torch(o, device) for o in jax_out)
        return jax_to_torch(jax_out, device)

    @staticmethod
    def backward(ctx, *grad_outputs):
        jax_grads_out = tuple(torch_to_jax(g) for g in grad_outputs)
        if len(jax_grads_out) == 1:
            jax_grads_out = jax_grads_out[0]

        jax_grads_in = ctx.vjp_fn(jax_grads_out)

        torch_grads = tuple(
            jax_to_torch(g, ctx.device) for g in jax_grads_in
        )

        # None for jax_run_fn, None for n_param_tensors, grads for params, None for others
        n_other = len(grad_outputs)  # conservative
        result = (None, None) + torch_grads
        return result


class JAXComponent(DifferentiableComponent, BMIMixin):
    """Base class for JAX-native model components.

    Wraps a JAX step function and optional batch (lax.scan) function.
    Parameters are held as PyTorch nn.Parameters with sigmoid transforms,
    and gradients flow through the JAXBridge autograd function.

    Args:
        name: Component identifier
        jax_step_fn: Per-timestep JAX function with signature
            ``(inputs, state, params, dt) -> (outputs, new_state)``
        jax_run_fn: Optional batch function (e.g. using lax.scan) with same
            signature but processing all timesteps at once
        param_specs: List of ParameterSpec for optimizable parameters
        state_size: Number of state variables
        input_fluxes: List of FluxSpec for inputs
        output_fluxes: List of FluxSpec for outputs
    """

    def __init__(
        self,
        name: str,
        jax_step_fn: Callable,
        jax_run_fn: Optional[Callable] = None,
        param_specs: Optional[List[ParameterSpec]] = None,
        state_size: int = 0,
        input_flux_specs: Optional[List[FluxSpec]] = None,
        output_flux_specs: Optional[List[FluxSpec]] = None,
    ):
        self._name = name
        self._jax_step = jax_step_fn
        self._jax_run = jax_run_fn
        self._param_specs = param_specs or []
        self._state_size = state_size
        self._input_fluxes = input_flux_specs or []
        self._output_fluxes = output_flux_specs or []
        self._state: Any = None
        self._last_outputs: Dict[str, Any] = {}
        self._config: dict = {}

        # Create PyTorch parameters (unconstrained space, sigmoid-transformed)
        self._raw_params = nn.ParameterList()
        self._param_lower = []
        self._param_upper = []
        for spec in self._param_specs:
            n = spec.n_spatial if spec.spatial and spec.n_spatial else 1
            init = torch.zeros(n)
            self._raw_params.append(nn.Parameter(init))
            self._param_lower.append(spec.lower_bound)
            self._param_upper.append(spec.upper_bound)

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return self._input_fluxes

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return self._output_fluxes

    @property
    def parameters(self) -> List[ParameterSpec]:
        return self._param_specs

    @property
    def gradient_method(self) -> GradientMethod:
        return GradientMethod.AUTOGRAD

    @property
    def state_size(self) -> int:
        return self._state_size

    @property
    def requires_batch(self) -> bool:
        return self._jax_run is not None

    def initialize(self, config: dict) -> None:
        self._config = config

    def get_initial_state(self) -> torch.Tensor:
        return torch.zeros(self._state_size)

    def get_physical_parameters(self) -> Dict[str, torch.Tensor]:
        result = {}
        for i, spec in enumerate(self._param_specs):
            raw = self._raw_params[i]
            lo, hi = self._param_lower[i], self._param_upper[i]
            phys = lo + (hi - lo) * torch.sigmoid(raw)
            result[spec.name] = phys
        return result

    def get_physical_params_dict(self) -> Dict[str, torch.Tensor]:
        """Return physical parameters as a flat dict (convenience)."""
        return self.get_physical_parameters()

    def _params_as_jax(self):
        """Convert PyTorch parameters to JAX-compatible dict."""
        _, jnp = _import_jax()
        params = {}
        for i, spec in enumerate(self._param_specs):
            raw = self._raw_params[i]
            lo, hi = self._param_lower[i], self._param_upper[i]
            phys = lo + (hi - lo) * torch.sigmoid(raw)
            params[spec.name] = phys
        return params

    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        state: torch.Tensor,
        dt: float,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Single timestep via JAXBridge."""
        params = self._params_as_jax()

        # Collect all differentiable tensors
        param_tensors = [params[s.name] for s in self._param_specs]
        input_tensors = [inputs[f.name] for f in self._input_fluxes if f.name in inputs]

        n_params = len(param_tensors)
        all_tensors = param_tensors + input_tensors + [state, torch.tensor(dt)]

        def jax_wrapper(*args):
            jax_params = {s.name: args[i] for i, s in enumerate(self._param_specs)}
            n_p = len(self._param_specs)
            jax_inputs = {
                f.name: args[n_p + j]
                for j, f in enumerate(self._input_fluxes)
                if f.name in inputs
            }
            jax_state = args[-2]
            jax_dt = args[-1]
            return self._jax_step(jax_inputs, jax_state, jax_params, jax_dt)

        result = JAXBridge.apply(jax_wrapper, n_params, *all_tensors)

        if isinstance(result, tuple) and len(result) == 2:
            outputs_tensor, new_state = result
        else:
            outputs_tensor = result
            new_state = state

        if isinstance(outputs_tensor, dict):
            output_dict = outputs_tensor
        else:
            output_dict = {self._output_fluxes[0].name: outputs_tensor}

        return output_dict, new_state

    def run(
        self,
        inputs: Dict[str, torch.Tensor],
        state: torch.Tensor,
        dt: float,
        n_timesteps: int,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Batch execution, using lax.scan path if available."""
        if self._jax_run is not None:
            params = self._params_as_jax()
            param_tensors = [params[s.name] for s in self._param_specs]
            input_tensors = [inputs[f.name] for f in self._input_fluxes if f.name in inputs]

            n_params = len(param_tensors)
            all_tensors = param_tensors + input_tensors + [
                state, torch.tensor(float(dt)), torch.tensor(float(n_timesteps))
            ]

            def jax_batch_wrapper(*args):
                jax_params = {s.name: args[i] for i, s in enumerate(self._param_specs)}
                n_p = len(self._param_specs)
                jax_inputs = {
                    f.name: args[n_p + j]
                    for j, f in enumerate(self._input_fluxes)
                    if f.name in inputs
                }
                jax_state = args[-3]
                jax_dt = args[-2]
                jax_n = args[-1]
                return self._jax_run(jax_inputs, jax_state, jax_params, jax_dt, int(jax_n))

            result = JAXBatchBridge.apply(jax_batch_wrapper, n_params, *all_tensors)

            if isinstance(result, tuple) and len(result) == 2:
                outputs_tensor, new_state = result
            else:
                outputs_tensor = result
                new_state = state

            if isinstance(outputs_tensor, dict):
                return outputs_tensor, new_state
            return {self._output_fluxes[0].name: outputs_tensor}, new_state

        # Fallback: step-by-step
        return super().run(inputs, state, dt, n_timesteps)

    def get_torch_parameters(self) -> List[nn.Parameter]:
        return list(self._raw_params)

    # -- BMI interface implementation --------------------------------------

    def bmi_initialize(self, config: dict) -> None:
        self.initialize(config)
        self._state = self.get_initial_state()

    def bmi_update(self, inputs: Dict[str, Any], dt: float) -> Dict[str, Any]:
        tensor_inputs = {
            k: torch.as_tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        if self._state is None:
            self._state = self.get_initial_state()
        outputs, self._state = self.step(tensor_inputs, self._state, dt)
        self._last_outputs = {k: v.detach() for k, v in outputs.items()}
        return {k: v.detach().numpy() for k, v in outputs.items()}

    def bmi_update_batch(
        self, inputs: Dict[str, Any], dt: float, n_timesteps: int
    ) -> Dict[str, Any]:
        tensor_inputs = {
            k: torch.as_tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        if self._state is None:
            self._state = self.get_initial_state()
        outputs, self._state = self.run(tensor_inputs, self._state, dt, n_timesteps)
        self._last_outputs = {k: v.detach() for k, v in outputs.items()}
        return {k: v.detach().numpy() for k, v in outputs.items()}

    def bmi_finalize(self) -> None:
        self._state = None
        self._last_outputs = {}

    def bmi_get_state(self) -> Any:
        return self._state

    def bmi_set_state(self, state: Any) -> None:
        self._state = state

    def bmi_get_value(self, name: str) -> Any:
        if name in self._last_outputs:
            return self._last_outputs[name].numpy()
        raise KeyError(f"Unknown variable '{name}'")

    def bmi_set_value(self, name: str, value: Any) -> None:
        pass  # Parameters set via PyTorch nn.Parameter

    def bmi_get_output_var_names(self) -> List[str]:
        return [f.name for f in self._output_fluxes]

    def bmi_get_input_var_names(self) -> List[str]:
        return [f.name for f in self._input_fluxes]
