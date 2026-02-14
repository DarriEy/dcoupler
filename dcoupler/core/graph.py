from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque

import torch

from .component import DifferentiableComponent, FluxSpec
from .connection import FluxConnection, SpatialRemapper
from .temporal import TemporalOrchestrator
from dcoupler.utils.temporal import interpolate_flux


class CouplingGraph:
    """
    Directed acyclic graph of coupled differentiable components.
    """

    def __init__(self, conservation_mode: Optional[str] = None, conservation_tolerance: float = 1e-6) -> None:
        self.components: Dict[str, DifferentiableComponent] = {}
        self.connections: List[FluxConnection] = []
        self._execution_order: Optional[List[str]] = None
        self._validated: bool = False
        self._conservation: Optional["ConservationChecker"] = None
        if conservation_mode is not None:
            from .conservation import ConservationChecker
            self._conservation = ConservationChecker(
                mode=conservation_mode, tolerance=conservation_tolerance
            )

    def add_component(self, component: DifferentiableComponent) -> None:
        name = component.name
        if name in self.components:
            raise ValueError(f"Component '{name}' already registered")
        self.components[name] = component
        self._execution_order = None
        self._validated = False

    def _get_flux_spec(self, component: DifferentiableComponent, flux_name: str, kind: str) -> FluxSpec:
        if kind == "input":
            fluxes = component.input_fluxes
        else:
            fluxes = component.output_fluxes
        for spec in fluxes:
            if spec.name == flux_name:
                return spec
        raise ValueError(f"Component '{component.name}' has no {kind} flux '{flux_name}'")

    def connect(
        self,
        source: str,
        source_flux: str,
        target: str,
        target_flux: str,
        spatial_remap: Optional[SpatialRemapper] = None,
        unit_conversion: Optional[float] = None,
        temporal_interp: str = "step",
    ) -> None:
        if source not in self.components:
            raise ValueError(f"Unknown source component '{source}'")
        if target not in self.components:
            raise ValueError(f"Unknown target component '{target}'")

        src_comp = self.components[source]
        tgt_comp = self.components[target]

        src_spec = self._get_flux_spec(src_comp, source_flux, "output")
        tgt_spec = self._get_flux_spec(tgt_comp, target_flux, "input")

        if src_spec.units != tgt_spec.units and unit_conversion is None:
            raise ValueError(
                f"Unit mismatch for {source}.{source_flux} ({src_spec.units}) -> "
                f"{target}.{target_flux} ({tgt_spec.units}); provide unit_conversion"
            )
        if src_spec.spatial_type != tgt_spec.spatial_type and spatial_remap is None:
            raise ValueError(
                f"Spatial mismatch for {source}.{source_flux} ({src_spec.spatial_type}) -> "
                f"{target}.{target_flux} ({tgt_spec.spatial_type}); provide spatial_remap"
            )

        connection = FluxConnection(
            source_component=source,
            source_flux=source_flux,
            target_component=target,
            target_flux=target_flux,
            spatial_remap=spatial_remap,
            unit_conversion=unit_conversion,
            temporal_interp=temporal_interp,
            conserved_quantity=src_spec.conserved_quantity or tgt_spec.conserved_quantity,
        )

        self.connections.append(connection)
        # Validate acyclicity
        try:
            self._execution_order = self._topological_sort()
        except ValueError:
            self.connections.pop()
            raise
        self._validated = False

    def _topological_sort(self) -> List[str]:
        in_degree = {name: 0 for name in self.components}
        adjacency = defaultdict(list)

        for conn in self.connections:
            adjacency[conn.source_component].append(conn.target_component)
            in_degree[conn.target_component] += 1

        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for nxt in adjacency[node]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(self.components):
            raise ValueError("Coupling graph contains a cycle")
        return order

    def validate(self) -> List[str]:
        warnings: List[str] = []
        if not self._execution_order:
            self._execution_order = self._topological_sort()

        connected_inputs = defaultdict(set)
        connected_outputs = defaultdict(set)

        for conn in self.connections:
            connected_inputs[conn.target_component].add(conn.target_flux)
            connected_outputs[conn.source_component].add(conn.source_flux)

        for comp in self.components.values():
            for flux in comp.input_fluxes:
                if flux.name not in connected_inputs[comp.name] and not flux.optional:
                    warnings.append(f"Unconnected required input: {comp.name}.{flux.name}")
            for flux in comp.output_fluxes:
                if flux.name not in connected_outputs[comp.name]:
                    warnings.append(f"Unconnected output: {comp.name}.{flux.name}")

        self._validated = True
        return warnings

    def _slice_time(self, tensor: torch.Tensor, t: int) -> torch.Tensor:
        if tensor.dim() == 0:
            return tensor
        if tensor.dim() >= 1:
            if tensor.shape[0] == 1:
                return tensor[0]
            if t < tensor.shape[0]:
                return tensor[t]
        return tensor

    def _apply_connection(
        self,
        conn: FluxConnection,
        tensor: torch.Tensor,
        source_areas: Optional[torch.Tensor] = None,
        target_areas: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        result = tensor
        if conn.spatial_remap is not None:
            result = conn.spatial_remap.forward(result)
        if conn.unit_conversion is not None:
            result = result * conn.unit_conversion
        # Conservation check/enforcement
        if self._conservation is not None and conn.conserved_quantity:
            s_areas = source_areas if source_areas is not None else torch.ones_like(tensor)
            t_areas = target_areas if target_areas is not None else torch.ones_like(result)
            connection_dt = dt if dt is not None else 1.0
            corrected = self._conservation.check_connection(
                conn, tensor, result, s_areas, t_areas, connection_dt
            )
            if corrected is not None:
                result = corrected
        return result

    def forward(
        self,
        external_inputs: Dict[str, Dict[str, torch.Tensor]],
        initial_states: Optional[Dict[str, torch.Tensor]] = None,
        n_timesteps: int = 1,
        dt: Optional[float] = None,
        temporal: Optional[TemporalOrchestrator] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        if dt is None:
            raise ValueError("dt must be provided")
        if not self._execution_order:
            self._execution_order = self._topological_sort()

        if temporal is not None:
            return self._forward_temporal(external_inputs, initial_states, n_timesteps, dt, temporal)

        needs_batch = any(self.components[name].requires_batch for name in self._execution_order)
        if needs_batch:
            return self._forward_batch(external_inputs, initial_states, n_timesteps, dt)
        return self._forward_stepwise(external_inputs, initial_states, n_timesteps, dt)

    def _forward_stepwise(
        self,
        external_inputs: Dict[str, Dict[str, torch.Tensor]],
        initial_states: Optional[Dict[str, torch.Tensor]],
        n_timesteps: int,
        dt: float,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        states: Dict[str, torch.Tensor] = {}
        for name, comp in self.components.items():
            if initial_states and name in initial_states:
                states[name] = initial_states[name]
            else:
                states[name] = comp.get_initial_state()

        outputs: Dict[str, Dict[str, List[torch.Tensor]]] = {
            name: {spec.name: [] for spec in comp.output_fluxes}
            for name, comp in self.components.items()
        }

        for t in range(n_timesteps):
            current_outputs: Dict[str, Dict[str, torch.Tensor]] = {}

            for name in self._execution_order:
                comp = self.components[name]
                inputs: Dict[str, torch.Tensor] = {}

                for flux in comp.input_fluxes:
                    values = []
                    for conn in self.connections:
                        if conn.target_component == name and conn.target_flux == flux.name:
                            src_out = current_outputs[conn.source_component][conn.source_flux]
                            values.append(self._apply_connection(conn, src_out))

                    if values:
                        inputs[flux.name] = torch.stack(values, dim=0).sum(dim=0)
                        continue

                    ext = external_inputs.get(name, {}).get(flux.name)
                    if ext is None:
                        if flux.optional:
                            continue
                        raise ValueError(
                            f"Missing input '{name}.{flux.name}' at timestep {t}"
                        )
                    inputs[flux.name] = self._slice_time(ext, t)

                step_outputs, new_state = comp.step(inputs, states[name], dt)
                states[name] = new_state
                current_outputs[name] = step_outputs
                for flux_name, tensor in step_outputs.items():
                    outputs[name][flux_name].append(tensor)

        stacked_outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, fluxes in outputs.items():
            stacked_outputs[name] = {
                flux_name: torch.stack(values, dim=0) if len(values) > 0 else torch.empty(0)
                for flux_name, values in fluxes.items()
            }
        return stacked_outputs

    def _forward_batch(
        self,
        external_inputs: Dict[str, Dict[str, torch.Tensor]],
        initial_states: Optional[Dict[str, torch.Tensor]],
        n_timesteps: int,
        dt: float,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        states: Dict[str, torch.Tensor] = {}
        for name, comp in self.components.items():
            if initial_states and name in initial_states:
                states[name] = initial_states[name]
            else:
                states[name] = comp.get_initial_state()

        outputs: Dict[str, Dict[str, torch.Tensor]] = {}

        for name in self._execution_order:
            comp = self.components[name]
            inputs: Dict[str, torch.Tensor] = {}

            for flux in comp.input_fluxes:
                values = []
                for conn in self.connections:
                    if conn.target_component == name and conn.target_flux == flux.name:
                        src_out = outputs[conn.source_component][conn.source_flux]
                        values.append(self._apply_connection(conn, src_out))
                if values:
                    inputs[flux.name] = torch.stack(values, dim=0).sum(dim=0)
                    continue

                ext = external_inputs.get(name, {}).get(flux.name)
                if ext is None:
                    if flux.optional:
                        continue
                    raise ValueError(f"Missing input '{name}.{flux.name}'")
                inputs[flux.name] = ext

            if hasattr(comp, "run") and callable(getattr(comp, "run")):
                comp_outputs, new_state = comp.run(inputs, states[name], dt, n_timesteps)
            else:
                comp_outputs, new_state = comp.run(inputs, states[name], dt, n_timesteps)
            states[name] = new_state
            outputs[name] = comp_outputs

        return outputs

    def _forward_temporal(
        self,
        external_inputs: Dict[str, Dict[str, torch.Tensor]],
        initial_states: Optional[Dict[str, torch.Tensor]],
        n_timesteps: int,
        dt: float,
        temporal: TemporalOrchestrator,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        states: Dict[str, torch.Tensor] = {}
        for name, comp in self.components.items():
            if initial_states and name in initial_states:
                states[name] = initial_states[name]
            else:
                states[name] = comp.get_initial_state()

        outputs: Dict[str, Dict[str, List[torch.Tensor]]] = {
            name: {spec.name: [] for spec in comp.output_fluxes}
            for name, comp in self.components.items()
        }

        for outer_t in range(n_timesteps):
            current_outputs: Dict[str, Dict[str, torch.Tensor]] = {}

            for name in self._execution_order:
                comp = self.components[name]
                comp_dt = temporal.get_component_dt(name)
                substeps = temporal.get_substeps(name)

                inputs: Dict[str, torch.Tensor] = {}
                for flux in comp.input_fluxes:
                    interp_method = "step"
                    values = []
                    for conn in self.connections:
                        if conn.target_component == name and conn.target_flux == flux.name:
                            src_out = current_outputs[conn.source_component][conn.source_flux]
                            values.append(self._apply_connection(conn, src_out))
                            interp_method = conn.temporal_interp or "step"

                    if values:
                        combined = torch.stack(values, dim=0).sum(dim=0)
                        interp = interpolate_flux(
                            combined,
                            interp_method,
                            substeps,
                        )
                        inputs[flux.name] = interp
                        continue

                    ext = external_inputs.get(name, {}).get(flux.name)
                    if ext is None:
                        if flux.optional:
                            continue
                        raise ValueError(
                            f"Missing input '{name}.{flux.name}' at timestep {outer_t}"
                        )
                    outer_slice = self._slice_time(ext, outer_t)
                    inputs[flux.name] = interpolate_flux(
                        outer_slice,
                        "step",
                        substeps,
                    )

                comp_outputs, new_state = comp.run(
                    inputs,
                    states[name],
                    comp_dt,
                    substeps,
                )
                states[name] = new_state
                current_outputs[name] = {k: v[-1] if v.dim() > 1 else v for k, v in comp_outputs.items()}
                for flux_name, tensor in comp_outputs.items():
                    outputs[name][flux_name].append(tensor)

        stacked_outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, fluxes in outputs.items():
            stacked_outputs[name] = {
                flux_name: torch.stack(values, dim=0) if len(values) > 0 else torch.empty(0)
                for flux_name, values in fluxes.items()
            }
        return stacked_outputs

    def get_all_parameters(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        for comp in self.components.values():
            params.extend(comp.get_torch_parameters())
        return params
