from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch

from dcoupler.core.component import ParameterSpec
from dcoupler.core.graph import CouplingGraph


class ParameterManager:
    """
    Manages optimizable parameters across all components.
    """

    def __init__(self, coupling_graph: CouplingGraph):
        self.graph = coupling_graph
        self.frozen: set = set()
        self._param_map = self._build_param_map()

    def _build_param_map(self) -> Dict[Tuple[str, str], torch.nn.Parameter]:
        mapping: Dict[Tuple[str, str], torch.nn.Parameter] = {}
        for comp in self.graph.components.values():
            specs = comp.parameters
            params = comp.get_torch_parameters()
            if len(specs) != len(params):
                raise ValueError(
                    f"Parameter mismatch for component '{comp.name}': "
                    f"{len(specs)} specs vs {len(params)} torch parameters"
                )
            for spec, param in zip(specs, params):
                mapping[(comp.name, spec.name)] = param
        return mapping

    def get_optimizer_params(self) -> List[dict]:
        """
        Return parameter groups for torch.optim.
        Supports per-component learning rates via optional `learning_rate` attribute.
        """
        groups: List[dict] = []
        for comp in self.graph.components.values():
            params = [p for p in comp.get_torch_parameters() if p.requires_grad]
            if not params:
                continue
            lr = getattr(comp, "learning_rate", None)
            group = {"params": params}
            if lr is not None:
                group["lr"] = lr
            groups.append(group)
        return groups

    def freeze(self, component: str, param_name: Optional[str] = None) -> None:
        if param_name is None:
            for (comp_name, name), param in self._param_map.items():
                if comp_name == component:
                    param.requires_grad = False
                    self.frozen.add((comp_name, name))
            return
        key = (component, param_name)
        param = self._param_map.get(key)
        if param is None:
            raise KeyError(f"Unknown parameter {component}.{param_name}")
        param.requires_grad = False
        self.frozen.add(key)

    def unfreeze(self, component: str, param_name: Optional[str] = None) -> None:
        if param_name is None:
            for (comp_name, name), param in self._param_map.items():
                if comp_name == component:
                    param.requires_grad = True
                    self.frozen.discard((comp_name, name))
            return
        key = (component, param_name)
        param = self._param_map.get(key)
        if param is None:
            raise KeyError(f"Unknown parameter {component}.{param_name}")
        param.requires_grad = True
        self.frozen.discard(key)

    def spatial_regularization_loss(self, strength: float = 1.0) -> torch.Tensor:
        """
        L2 penalty on spatial variance of parameters.
        Encourages spatial smoothness. Differentiable.
        """
        if strength <= 0:
            return torch.tensor(0.0)

        total = torch.tensor(0.0)
        for comp in self.graph.components.values():
            phys = comp.get_physical_parameters()
            for spec in comp.parameters:
                if not spec.spatial:
                    continue
                if spec.name not in phys:
                    continue
                param = phys[spec.name]
                if param.dim() == 0 or param.shape[0] < 2:
                    continue
                denom = max(spec.upper_bound - spec.lower_bound, 1e-12)
                normalized = (param - spec.lower_bound) / denom
                total = total + normalized.var(dim=0).mean()
        return total * strength

    def save_checkpoint(self, path: str) -> None:
        state: Dict[str, Dict] = {"components": {}, "frozen": list(self.frozen)}
        for name, comp in self.graph.components.items():
            if hasattr(comp, "state_dict") and callable(getattr(comp, "state_dict")):
                state["components"][name] = {"state_dict": comp.state_dict()}
            else:
                params = {}
                for spec in comp.parameters:
                    key = (name, spec.name)
                    param = self._param_map.get(key)
                    if param is not None:
                        params[spec.name] = param.detach().cpu()
                state["components"][name] = {"params": params}
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        frozen = state.get("frozen", [])
        self.frozen = set(tuple(x) for x in frozen)

        for name, comp_state in state.get("components", {}).items():
            comp = self.graph.components.get(name)
            if comp is None:
                continue
            if "state_dict" in comp_state and hasattr(comp, "load_state_dict"):
                comp.load_state_dict(comp_state["state_dict"], strict=False)
            elif "params" in comp_state:
                for param_name, tensor in comp_state["params"].items():
                    key = (name, param_name)
                    param = self._param_map.get(key)
                    if param is not None:
                        param.data = tensor.to(param.device)

        # Reapply frozen flags
        for (comp_name, param_name), param in self._param_map.items():
            if (comp_name, param_name) in self.frozen:
                param.requires_grad = False
