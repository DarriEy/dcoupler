from __future__ import annotations

from typing import Dict

import torch

from dcoupler.core.graph import CouplingGraph


class GradientDiagnostics:
    """Verify gradient flow through the coupled system."""

    @staticmethod
    def _scalarize_outputs(outputs: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        total = torch.tensor(0.0)
        for comp_outputs in outputs.values():
            for tensor in comp_outputs.values():
                if tensor.numel() == 0:
                    continue
                total = total + tensor.sum()
        return total

    @staticmethod
    def verify_gradient_flow(
        coupling_graph: CouplingGraph,
        external_inputs: Dict[str, Dict[str, torch.Tensor]],
        n_test_steps: int = 10,
        dt: float = 1.0,
    ) -> Dict[str, Dict]:
        params = coupling_graph.get_all_parameters()
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        outputs = coupling_graph.forward(
            external_inputs=external_inputs,
            n_timesteps=n_test_steps,
            dt=dt,
        )
        loss = GradientDiagnostics._scalarize_outputs(outputs)
        loss.backward()

        results: Dict[str, Dict] = {}
        for comp in coupling_graph.components.values():
            comp_results = {}
            param_specs = comp.parameters
            param_tensors = comp.get_torch_parameters()
            for spec, tensor in zip(param_specs, param_tensors):
                grad = tensor.grad
                if grad is None:
                    comp_results[spec.name] = {
                        "has_gradient": False,
                        "grad_norm": 0.0,
                        "grad_max": 0.0,
                        "grad_min": 0.0,
                    }
                else:
                    comp_results[spec.name] = {
                        "has_gradient": bool(torch.any(grad != 0)),
                        "grad_norm": float(torch.norm(grad).item()),
                        "grad_max": float(grad.max().item()),
                        "grad_min": float(grad.min().item()),
                    }
            results[comp.name] = comp_results
        return results

    @staticmethod
    def gradient_check_numerical(
        coupling_graph: CouplingGraph,
        external_inputs: Dict[str, Dict[str, torch.Tensor]],
        component: str,
        param_name: str,
        epsilon: float = 1e-4,
        n_test_steps: int = 5,
        dt: float = 1.0,
    ) -> Dict:
        comp = coupling_graph.components[component]
        spec_map = {spec.name: i for i, spec in enumerate(comp.parameters)}
        if param_name not in spec_map:
            raise KeyError(f"Unknown parameter {component}.{param_name}")
        idx = spec_map[param_name]
        param = comp.get_torch_parameters()[idx]

        for p in coupling_graph.get_all_parameters():
            if p.grad is not None:
                p.grad.zero_()

        outputs = coupling_graph.forward(
            external_inputs=external_inputs,
            n_timesteps=n_test_steps,
            dt=dt,
        )
        loss = GradientDiagnostics._scalarize_outputs(outputs)
        loss.backward()
        ad_grad = param.grad.detach().mean().item() if param.grad is not None else 0.0

        with torch.no_grad():
            base = param.data.clone()
            param.data = base + epsilon
            outputs_plus = coupling_graph.forward(
                external_inputs=external_inputs,
                n_timesteps=n_test_steps,
                dt=dt,
            )
            loss_plus = GradientDiagnostics._scalarize_outputs(outputs_plus).item()

            param.data = base - epsilon
            outputs_minus = coupling_graph.forward(
                external_inputs=external_inputs,
                n_timesteps=n_test_steps,
                dt=dt,
            )
            loss_minus = GradientDiagnostics._scalarize_outputs(outputs_minus).item()

            param.data = base

        fd_grad = (loss_plus - loss_minus) / (2 * epsilon)
        rel_error = abs(ad_grad - fd_grad) / (abs(fd_grad) + 1e-12)

        return {
            "ad_gradient": ad_grad,
            "fd_gradient": fd_grad,
            "relative_error": rel_error,
            "match": rel_error < 1e-3,
        }
