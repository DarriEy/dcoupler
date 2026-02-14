# dCoupler: Differentiable Earth System Coupler

## Specification Document v0.1

**Author:** Darri Eythorsson  
**Date:** February 2026  
**Status:** Draft specification for code generation

---

## 1. Purpose and Scope

dCoupler is a standalone Python library for coupling differentiable Earth system model components with end-to-end gradient propagation. It provides the infrastructure to wire together heterogeneous physics components (land surface, routing, groundwater, atmosphere, etc.) so that gradients flow seamlessly across component boundaries through the full coupled system.

### 1.1 What dCoupler Is

- A lightweight coupling library with minimal dependencies (PyTorch, numpy, xarray)
- A component registry with declared flux interfaces
- A coupling graph that manages flux exchange, spatial remapping, and temporal interpolation — all differentiable
- A multi-observation optimization engine
- Framework-agnostic: components can be JAX-native, PyTorch-native, C++ with Enzyme AD, or black-box with finite-difference Jacobians

### 1.2 What dCoupler Is Not

- Not a workflow orchestrator (that's SYMFLUENCE)
- Not a data acquisition pipeline
- Not an HPC job manager
- Not a model itself — it couples models

### 1.3 Design Principles

1. **Gradients are first-class citizens.** Every operation in the coupler (spatial remapping, temporal interpolation, conservation correction, flux aggregation) must be differentiable. If it breaks gradient flow, it doesn't belong in dCoupler.

2. **Components are black boxes with declared interfaces.** dCoupler doesn't care what's inside a component. It only cares about: what fluxes go in, what fluxes come out, at what spatial and temporal resolution, and how to get Jacobians.

3. **Standalone simplicity.** A researcher must be able to `pip install dcoupler`, write a 50-line script coupling two simple models, and run gradient-based calibration on a laptop. No YAML, no HPC, no configuration ceremony.

4. **Zero overhead for simple cases.** Coupling two components through dCoupler must produce identical gradients and equivalent performance to hand-written coupling (cf. DODO's CoupledFUSERoute). The abstraction must not cost accuracy or speed.

5. **Conservation by construction.** Mass, energy, and momentum must be conserved across coupling interfaces. Conservation constraints are themselves differentiable.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    User Script                       │
│  (defines components, wiring, observations, runs     │
│   optimization or forward prediction)                │
├─────────────────────────────────────────────────────┤
│                  Optimization Engine                 │
│  (multi-observation loss, parameter management,      │
│   gradient-based training loop)                      │
├─────────────────────────────────────────────────────┤
│                   Coupling Graph                     │
│  (topological sort of components, temporal           │
│   orchestration, operator splitting)                 │
├──────────┬──────────┬──────────┬────────────────────┤
│  Flux    │ Spatial  │ Temporal │  Conservation      │
│  Conn.   │ Remap    │ Interp   │  Enforcement       │
├──────────┴──────────┴──────────┴────────────────────┤
│              Component Registry                      │
│  (DifferentiableComponent protocol, wrappers for     │
│   PyTorch, JAX, C++/Enzyme, black-box)               │
└─────────────────────────────────────────────────────┘
```

---

## 3. Component Protocol

### 3.1 DifferentiableComponent (Abstract Base)

Every model component implements this protocol. It is deliberately minimal.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import torch


class FluxDirection(Enum):
    INPUT = "input"
    OUTPUT = "output"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class FluxSpec:
    """Declaration of a single flux variable at a component boundary."""
    name: str                          # e.g. "precipitation", "runoff", "latent_heat"
    units: str                         # e.g. "mm/day", "m3/s", "W/m2"
    direction: FluxDirection
    spatial_type: str                  # "hru", "reach", "grid", "point"
    temporal_resolution: float         # seconds (e.g. 86400 for daily, 3600 for hourly)
    dims: Tuple[str, ...]             # dimension names, e.g. ("time", "hru")
    optional: bool = False             # whether this flux can be absent
    conserved_quantity: Optional[str] = None  # e.g. "water_mass", "energy"


@dataclass
class ParameterSpec:
    """Declaration of an optimizable parameter."""
    name: str
    lower_bound: float
    upper_bound: float
    spatial: bool = False              # True if parameter varies spatially
    n_spatial: Optional[int] = None    # number of spatial units if spatial=True
    log_transform: bool = False        # optimize in log-space


class GradientMethod(Enum):
    """How this component provides gradients."""
    AUTOGRAD = "autograd"              # PyTorch/JAX native autograd
    ENZYME = "enzyme"                  # Enzyme AD (C++ reverse-mode)
    ADJOINT = "adjoint"                # Adjoint model (continuous or discrete)
    FINITE_DIFFERENCE = "finite_diff"  # Numerical finite differences
    NONE = "none"                      # Forward-only, no gradients


class DifferentiableComponent:
    """
    Protocol for a differentiable model component.
    
    Implementations wrap specific models (FUSE, SUMMA, dRoute, dGW, etc.)
    and expose a uniform interface for the coupling graph.
    """
    
    @property
    def name(self) -> str:
        """Unique component identifier."""
        raise NotImplementedError
    
    @property
    def input_fluxes(self) -> List[FluxSpec]:
        """Fluxes this component requires as input."""
        raise NotImplementedError
    
    @property
    def output_fluxes(self) -> List[FluxSpec]:
        """Fluxes this component produces as output."""
        raise NotImplementedError
    
    @property
    def parameters(self) -> List[ParameterSpec]:
        """Optimizable parameters exposed by this component."""
        raise NotImplementedError
    
    @property
    def gradient_method(self) -> GradientMethod:
        """How this component provides gradients."""
        raise NotImplementedError
    
    @property
    def state_size(self) -> int:
        """Number of internal state variables."""
        raise NotImplementedError
    
    def initialize(self, config: dict) -> None:
        """One-time setup (load data, allocate memory, etc.)."""
        raise NotImplementedError
    
    def get_initial_state(self) -> torch.Tensor:
        """Return initial state tensor [n_spatial, n_states]."""
        raise NotImplementedError
    
    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        state: torch.Tensor,
        dt: float
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Advance component by one timestep.
        
        Args:
            inputs: Dict mapping flux name -> tensor
            state: Current state tensor [n_spatial, n_states]
            dt: Timestep in seconds
            
        Returns:
            outputs: Dict mapping flux name -> tensor
            new_state: Updated state tensor [n_spatial, n_states]
        """
        raise NotImplementedError
    
    def get_torch_parameters(self) -> List[torch.nn.Parameter]:
        """Return list of optimizable PyTorch parameters."""
        raise NotImplementedError
    
    def get_physical_parameters(self) -> Dict[str, torch.Tensor]:
        """Return current physical parameter values (after transforms)."""
        raise NotImplementedError
```

### 3.2 Component Wrapper Examples

The library ships with base wrapper classes for common backends:

```python
class PyTorchComponent(DifferentiableComponent):
    """Wrapper for models already implemented as torch.nn.Module."""
    gradient_method = GradientMethod.AUTOGRAD

class JAXComponent(DifferentiableComponent):
    """
    Wrapper for JAX-native models (jSUMMA, jFUSE, dLand).
    Uses jax2torch or custom autograd.Function bridge.
    """
    gradient_method = GradientMethod.AUTOGRAD

class EnzymeComponent(DifferentiableComponent):
    """
    Wrapper for C++ models with Enzyme AD.
    Uses torch.autograd.Function with numpy<->torch bridge.
    Pattern extracted from DODO's DifferentiableRouting.
    """
    gradient_method = GradientMethod.ENZYME

class AdjointComponent(DifferentiableComponent):
    """
    Wrapper for models with adjoint capabilities.
    (e.g., MITgcm, CVODES-based solvers)
    """
    gradient_method = GradientMethod.ADJOINT

class BlackBoxComponent(DifferentiableComponent):
    """
    Wrapper for models without any AD support.
    Computes Jacobians via finite differences.
    Expensive but universal.
    """
    gradient_method = GradientMethod.FINITE_DIFFERENCE
```

---

## 4. Flux Connections

### 4.1 FluxConnection

A directed edge in the coupling graph that connects one component's output to another's input, with optional spatial remapping and unit conversion.

```python
@dataclass
class FluxConnection:
    """
    A differentiable connection between two component flux ports.
    
    Handles:
    - Unit conversion (e.g., mm/day -> m3/s)
    - Spatial remapping (e.g., HRU grid -> reach network)
    - Temporal interpolation (if components run at different dt)
    """
    source_component: str              # name of source component
    source_flux: str                   # name of output flux on source
    target_component: str              # name of target component
    target_flux: str                   # name of input flux on target
    
    spatial_remap: Optional['SpatialRemapper'] = None
    unit_conversion: Optional[float] = None   # multiplicative factor
    temporal_interp: Optional[str] = None      # "linear", "step", "conservative"
    
    conserved_quantity: Optional[str] = None   # for conservation checking


class SpatialRemapper:
    """
    Differentiable spatial remapping between component grids.
    
    All remapping methods produce a sparse matrix M such that:
        target_flux = M @ source_flux
    
    M is stored as a torch sparse tensor so matmul is differentiable.
    """
    
    @staticmethod
    def from_mapping_table(
        source_ids: np.ndarray,
        target_ids: np.ndarray, 
        hru_to_seg: np.ndarray,
        areas: Optional[np.ndarray] = None,
        unit_factor: float = 1.0
    ) -> 'SpatialRemapper':
        """
        Build remapper from explicit ID mapping.
        (Generalization of DODO's _build_mapping_matrix)
        
        This handles the common case: HRU runoff -> reach lateral inflow.
        """
        ...
    
    @staticmethod
    def from_grid_overlap(
        source_grid: 'GridSpec',
        target_grid: 'GridSpec',
        method: str = "conservative"  # or "bilinear", "nearest"
    ) -> 'SpatialRemapper':
        """
        Build remapper from grid overlap weights.
        Used for atmosphere-grid <-> land-grid remapping.
        Conservative remapping preserves integrals.
        """
        ...
    
    @staticmethod
    def identity(n: int) -> 'SpatialRemapper':
        """Identity remapper (same grid, no transformation)."""
        ...
    
    def forward(self, flux: torch.Tensor) -> torch.Tensor:
        """Apply remapping. Differentiable via sparse matmul."""
        return torch.sparse.mm(self.matrix, flux)
```

---

## 5. Coupling Graph

### 5.1 CouplingGraph

The central orchestrator. Manages the DAG of components and flux connections, handles temporal stepping with operator splitting, and provides the top-level `forward()` that the optimization engine calls.

```python
class CouplingGraph:
    """
    Directed acyclic graph of coupled differentiable components.
    
    Manages:
    - Component registration and validation
    - Flux connection wiring with spatial/temporal transforms
    - Topological ordering for forward execution
    - Temporal orchestration (operator splitting)
    - Conservation verification
    - End-to-end forward pass producing observables
    """
    
    def __init__(self):
        self.components: Dict[str, DifferentiableComponent] = {}
        self.connections: List[FluxConnection] = []
        self._execution_order: Optional[List[str]] = None
        self._validated: bool = False
    
    def add_component(self, component: DifferentiableComponent) -> None:
        """Register a component."""
        ...
    
    def connect(
        self,
        source: str, source_flux: str,
        target: str, target_flux: str,
        spatial_remap: Optional[SpatialRemapper] = None,
        unit_conversion: Optional[float] = None,
        temporal_interp: str = "step"
    ) -> None:
        """
        Add a flux connection between two components.
        
        Validates:
        - Source has the named output flux
        - Target has the named input flux
        - Units are compatible (or conversion is provided)
        - No cycles in the coupling graph
        """
        ...
    
    def validate(self) -> List[str]:
        """
        Validate the coupling graph. Returns list of warnings.
        
        Checks:
        - All required input fluxes have connections
        - No unconnected output fluxes (warning, not error)
        - No cycles
        - Spatial dimensions are compatible across connections
        - Conservation quantities balance
        """
        ...
    
    def forward(
        self,
        external_inputs: Dict[str, Dict[str, torch.Tensor]],
        initial_states: Optional[Dict[str, torch.Tensor]] = None,
        n_timesteps: int = 1,
        dt: Optional[float] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Run the full coupled system forward.
        
        Args:
            external_inputs: {component_name: {flux_name: tensor[time, spatial]}}
                Forcing data not produced by other components.
            initial_states: {component_name: state_tensor}
            n_timesteps: Number of outer coupling timesteps
            dt: Outer coupling timestep (seconds). 
                Components with finer resolution substep internally.
        
        Returns:
            all_outputs: {component_name: {flux_name: tensor[time, spatial]}}
                All output fluxes from all components, for observation operators.
        """
        ...
    
    def get_all_parameters(self) -> List[torch.nn.Parameter]:
        """Collect optimizable parameters from all components."""
        params = []
        for comp in self.components.values():
            params.extend(comp.get_torch_parameters())
        return params


class TemporalOrchestrator:
    """
    Manages operator splitting when components have different timesteps.
    
    Strategy: The coupling graph has an "outer timestep" (the coarsest).
    Components with finer timesteps substep internally. Fluxes exchanged
    at the outer timestep are interpolated for inner steps.
    
    All interpolation is differentiable (linear interp, step functions,
    or conservative redistribution).
    
    Example:
        Outer dt = 1 day (86400s)
        Land surface: 1 hour (substeps 24x within outer step)
        Routing: 15 min (substeps 96x within outer step)
        Atmosphere: 6 hours (substeps 4x within outer step)
        
    Flux exchange happens at outer boundaries.
    Inner steps use interpolated fluxes.
    """
    
    def __init__(self, outer_dt: float):
        self.outer_dt = outer_dt
        self.component_dt: Dict[str, float] = {}
    
    def set_component_dt(self, component: str, dt: float) -> None:
        """Set a component's internal timestep."""
        if self.outer_dt % dt != 0 and dt % self.outer_dt != 0:
            raise ValueError(
                f"Component dt ({dt}s) must evenly divide or be a multiple "
                f"of outer dt ({self.outer_dt}s)"
            )
        self.component_dt[component] = dt
    
    def get_substeps(self, component: str) -> int:
        """How many internal steps this component takes per outer step."""
        comp_dt = self.component_dt.get(component, self.outer_dt)
        if comp_dt <= self.outer_dt:
            return int(self.outer_dt / comp_dt)
        return 1  # component is coarser than outer, runs once
```

---

## 6. Conservation Enforcement

```python
class ConservationChecker:
    """
    Verifies and optionally enforces conservation across coupling interfaces.
    
    For each conserved quantity (water mass, energy), checks that:
        sum(source fluxes * dt * area) == sum(target fluxes * dt * area)
    
    Two modes:
    1. CHECK: Log warnings when conservation error exceeds threshold
    2. ENFORCE: Apply differentiable correction to close the budget
    
    Enforcement uses a differentiable projection:
        corrected_flux = flux * (source_total / target_total)
    This multiplicative correction preserves spatial patterns while
    closing the budget, and its Jacobian is well-defined.
    """
    
    def __init__(self, mode: str = "check", tolerance: float = 1e-6):
        self.mode = mode       # "check" or "enforce"
        self.tolerance = tolerance
        self.conservation_log: List[Dict] = []
    
    def check_connection(
        self,
        connection: FluxConnection,
        source_flux: torch.Tensor,
        target_flux: torch.Tensor,
        source_areas: torch.Tensor,
        target_areas: torch.Tensor,
        dt: float
    ) -> Optional[torch.Tensor]:
        """
        Check/enforce conservation for a single connection.
        
        Returns corrected target_flux if mode="enforce", else None.
        """
        ...
```

---

## 7. Observation Operators

### 7.1 ObservationOperator (Abstract)

Maps from model state/flux space to observation space. Must be differentiable.

```python
class ObservationOperator:
    """
    Differentiable mapping from model space to observation space.
    
    Each operator takes model outputs (fluxes, states) and produces
    a quantity directly comparable to observations.
    """
    
    @property
    def name(self) -> str:
        raise NotImplementedError
    
    @property
    def required_model_outputs(self) -> List[Tuple[str, str]]:
        """List of (component_name, flux_name) this operator needs."""
        raise NotImplementedError
    
    @property
    def observation_units(self) -> str:
        raise NotImplementedError
    
    def forward(
        self,
        model_outputs: Dict[str, Dict[str, torch.Tensor]],
        observation_metadata: dict
    ) -> torch.Tensor:
        """
        Compute predicted observation from model outputs.
        Must be differentiable w.r.t. model_outputs.
        """
        raise NotImplementedError


class StreamflowObserver(ObservationOperator):
    """
    Extract simulated discharge at gauge locations.
    Trivial operator: just indexes into routing output at gauge reaches.
    """
    name = "streamflow"
    observation_units = "m3/s"
    
    def __init__(self, gauge_reach_ids: List[int]):
        self.gauge_reach_ids = gauge_reach_ids
    
    @property
    def required_model_outputs(self):
        return [("routing", "discharge")]
    
    def forward(self, model_outputs, observation_metadata):
        discharge = model_outputs["routing"]["discharge"]
        # Index at gauge locations - differentiable indexing
        return discharge[:, self.gauge_indices]


class GRACEObserver(ObservationOperator):
    """
    Total water storage anomaly integrated over basin,
    convolved with GRACE averaging kernel.
    
    TWS = soil_water + groundwater + snow_water + surface_water
    Anomaly = TWS - TWS_mean (over reference period)
    GRACE_obs = kernel * TWS_anomaly  (spatial convolution)
    
    All operations differentiable.
    """
    name = "grace_tws"
    observation_units = "cm_water_equivalent"
    
    @property
    def required_model_outputs(self):
        return [
            ("land", "soil_moisture"),
            ("land", "snow_water_equivalent"),
            ("groundwater", "water_table_storage"),
            ("routing", "channel_storage"),
        ]


class SoilMoistureObserver(ObservationOperator):
    """
    Microwave brightness temperature forward model.
    
    Or simpler: direct comparison to retrieved soil moisture products,
    with appropriate depth/footprint matching.
    """
    name = "soil_moisture"


class EddyCovarianceObserver(ObservationOperator):
    """
    Compare modeled latent/sensible heat flux to flux tower observations.
    Handles footprint weighting if provided.
    """
    name = "eddy_covariance"
    
    @property 
    def required_model_outputs(self):
        return [
            ("land", "latent_heat_flux"),
            ("land", "sensible_heat_flux"),
        ]
```

---

## 8. Optimization Engine

### 8.1 Multi-Observation Loss

```python
class MultiObservationLoss:
    """
    Aggregates loss across multiple observation types and locations.
    
    Each observation source has:
    - An ObservationOperator (maps model -> obs space)
    - Observed data tensor
    - A loss function (from the loss library)
    - A weight
    
    Total loss = sum(weight_i * loss_i(H_i(model), obs_i))
    where H_i is the observation operator.
    """
    
    def __init__(self):
        self.terms: List[LossTerm] = []
    
    def add_term(
        self,
        observer: ObservationOperator,
        observed: torch.Tensor,
        loss_fn: str = "nse",           # from loss library
        weight: float = 1.0,
        warmup_steps: int = 0,
        mask: Optional[torch.Tensor] = None
    ) -> None:
        ...
    
    def compute(
        self,
        model_outputs: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and per-term diagnostics.
        
        Returns:
            total_loss: Scalar tensor (differentiable)
            diagnostics: {term_name: loss_value} for logging
        """
        ...


@dataclass
class LossTerm:
    observer: ObservationOperator
    observed: torch.Tensor
    loss_fn: callable
    weight: float
    warmup_steps: int
    mask: Optional[torch.Tensor]
```

### 8.2 Loss Function Library

Migrated from DODO, generalized to multi-site:

```python
# dcoupler/losses.py

def nse_loss(sim, obs, mask=None): ...
def log_nse_loss(sim, obs, mask=None, epsilon=0.1): ...
def kge_loss(sim, obs, mask=None): ...
def combined_nse_loss(sim, obs, mask=None, alpha=0.5): ...
def flow_duration_loss(sim, obs, mask=None): ...
def asymmetric_nse_loss(sim, obs, mask=None, under_weight=2.0): ...
def peak_weighted_nse(sim, obs, mask=None, quantile=0.9): ...
def triple_objective_loss(sim, obs, mask=None, w_nse=0.4, w_log=0.3, w_peak=0.3): ...
def rmse_loss(sim, obs, mask=None): ...
def mse_loss(sim, obs, mask=None): ...

# Multi-site versions: compute per-site, then aggregate
def multi_site_nse(sim, obs, mask=None, aggregation="mean"): ...
def multi_site_kge(sim, obs, mask=None, aggregation="mean"): ...
```

### 8.3 Parameter Manager

```python
class ParameterManager:
    """
    Manages optimizable parameters across all components.
    
    Responsibilities:
    - Collect parameters from all components
    - Apply transforms (sigmoid for bounded, log for positive)
    - Spatial regularization
    - Parameter freezing/unfreezing for staged calibration
    - Checkpoint save/restore
    """
    
    def __init__(self, coupling_graph: CouplingGraph):
        self.graph = coupling_graph
        self.frozen: set = set()       # parameter names that are frozen
    
    def get_optimizer_params(self) -> List[dict]:
        """
        Return parameter groups for torch.optim.
        Supports per-component learning rates.
        """
        ...
    
    def freeze(self, component: str, param_name: Optional[str] = None): ...
    def unfreeze(self, component: str, param_name: Optional[str] = None): ...
    
    def spatial_regularization_loss(self, strength: float = 1.0) -> torch.Tensor:
        """
        L2 penalty on spatial variance of parameters.
        Encourages spatial smoothness. Differentiable.
        (Migrated from DODO's train_model)
        """
        ...
    
    def save_checkpoint(self, path: str): ...
    def load_checkpoint(self, path: str): ...


class Trainer:
    """
    Training loop for coupled model optimization.
    
    Migrated and generalized from DODO's train_model().
    """
    
    def __init__(
        self,
        coupling_graph: CouplingGraph,
        loss: MultiObservationLoss,
        param_manager: ParameterManager,
        optimizer: str = "adam",
        lr: float = 0.01,
        scheduler: str = "warm_restarts",
        grad_clip: float = 1.0,
        n_epochs: int = 100,
    ):
        ...
    
    def train(
        self,
        external_inputs: Dict[str, Dict[str, torch.Tensor]],
        n_timesteps: int,
        dt: float,
        verbose: bool = True,
        checkpoint_every: int = 50,
        checkpoint_dir: Optional[str] = None
    ) -> TrainingResult:
        """
        Run the optimization loop.
        
        Each epoch:
        1. Forward pass through coupling graph
        2. Compute multi-observation loss
        3. Add regularization
        4. Backward pass (gradients flow through entire coupled system)
        5. Optimizer step with gradient clipping
        6. Scheduler step
        7. Log diagnostics
        """
        ...


@dataclass
class TrainingResult:
    history: Dict[str, List[float]]    # per-epoch metrics
    best_loss: float
    best_epoch: int
    final_parameters: Dict[str, Dict[str, torch.Tensor]]
    convergence_info: Dict
```

---

## 9. Diagnostics

```python
class GradientDiagnostics:
    """
    Verify gradient flow through the coupled system.
    Migrated from DODO's gradient verification logic.
    """
    
    @staticmethod
    def verify_gradient_flow(
        coupling_graph: CouplingGraph,
        external_inputs: Dict[str, Dict[str, torch.Tensor]],
        n_test_steps: int = 10
    ) -> Dict[str, Dict]:
        """
        Run a short forward+backward pass and check that every
        registered parameter has nonzero gradients.
        
        Returns:
            {component_name: {
                param_name: {
                    "has_gradient": bool,
                    "grad_norm": float,
                    "grad_max": float,
                    "grad_min": float,
                }
            }}
        """
        ...
    
    @staticmethod
    def gradient_check_numerical(
        coupling_graph: CouplingGraph,
        external_inputs: Dict,
        component: str,
        param_name: str,
        epsilon: float = 1e-4
    ) -> Dict:
        """
        Compare AD gradients to finite-difference gradients for a specific parameter.
        Essential for validating Enzyme/adjoint component wrappers.
        
        Returns:
            {
                "ad_gradient": float,
                "fd_gradient": float, 
                "relative_error": float,
                "match": bool (relative_error < 1e-3)
            }
        """
        ...


class ConservationDiagnostics:
    """Check mass/energy conservation across coupling interfaces."""
    
    @staticmethod
    def water_balance(
        coupling_graph: CouplingGraph,
        outputs: Dict[str, Dict[str, torch.Tensor]],
        dt: float
    ) -> Dict:
        """
        Compute water balance closure error.
        P - ET - Q - dS/dt should equal zero.
        """
        ...
```

---

## 10. Package Structure

```
dcoupler/
├── __init__.py                    # Public API exports
├── core/
│   ├── __init__.py
│   ├── component.py               # DifferentiableComponent protocol, FluxSpec, etc.
│   ├── connection.py              # FluxConnection, SpatialRemapper
│   ├── graph.py                   # CouplingGraph
│   ├── temporal.py                # TemporalOrchestrator
│   └── conservation.py            # ConservationChecker
├── wrappers/
│   ├── __init__.py
│   ├── pytorch.py                 # PyTorchComponent base
│   ├── jax.py                     # JAXComponent base (jax2torch bridge)
│   ├── enzyme.py                  # EnzymeComponent base (autograd.Function pattern)
│   ├── adjoint.py                 # AdjointComponent base
│   └── blackbox.py                # BlackBoxComponent (finite diff)
├── observers/
│   ├── __init__.py
│   ├── base.py                    # ObservationOperator ABC
│   ├── streamflow.py              # StreamflowObserver
│   ├── grace.py                   # GRACEObserver  
│   ├── soil_moisture.py           # SoilMoistureObserver
│   └── eddy_covariance.py         # EddyCovarianceObserver
├── losses/
│   ├── __init__.py
│   └── hydrological.py            # NSE, KGE, FDC, etc. (from DODO)
├── optimization/
│   ├── __init__.py
│   ├── parameters.py              # ParameterManager
│   ├── trainer.py                 # Trainer, TrainingResult
│   └── schedulers.py              # LR scheduler configs
├── diagnostics/
│   ├── __init__.py
│   ├── gradients.py               # GradientDiagnostics
│   └── conservation.py            # ConservationDiagnostics
└── utils/
    ├── __init__.py
    ├── units.py                   # Unit conversion registry
    └── io.py                      # Common I/O helpers (NetCDF, etc.)
```

---

## 11. DODO Migration Map

Explicit mapping from current DODO code to dCoupler components:

| DODO code | dCoupler destination | Notes |
|-----------|---------------------|-------|
| `DifferentiableRouting` | `wrappers/enzyme.py` | Generalize as `EnzymeComponent` template |
| `DifferentiableRoutingNative` | `wrappers/enzyme.py` | Second template for native-AD routers |
| `DifferentiableRoutingSaintVenant` | Router component impl | SVE-specific wrapper |
| `DifferentiableRoutingSaintVenantEnzyme` | Router component impl | SVE+Enzyme wrapper |
| `DifferentiableFUSEBatch` | Land component impl | FUSE-specific wrapper |
| `CoupledFUSERoute.__init__` | User script + `CouplingGraph` | Graph construction |
| `CoupledFUSERoute.forward` | `CouplingGraph.forward` | Generalized coupling |
| `_build_mapping_matrix` | `SpatialRemapper.from_mapping_table` | Generalized remapping |
| `_load_network` | External (SYMFLUENCE) | Not in dCoupler |
| `nse_loss`, `kge_loss`, etc. | `losses/hydrological.py` | Direct migration |
| `train_model` | `optimization/trainer.py` | Generalized training loop |
| `get_physical_params` + sigmoid | `optimization/parameters.py` | Bounded param transforms |
| Gradient verification in `main()` | `diagnostics/gradients.py` | Systematic diagnostics |
| `load_data_hourly` | External (SYMFLUENCE) | Not in dCoupler |
| All plotting functions | External (SYMFLUENCE) | Not in dCoupler |
| `resolve_data_path`, CLI | External (SYMFLUENCE) | Not in dCoupler |
| `ROUTING_METHOD_INFO` dispatch | Component registry pattern | Generalized dispatch |

---

## 12. Validation Criteria

The implementation is correct when:

### 12.1 Equivalence Test
A dCoupler configuration that wires FUSE + Muskingum-Cunge routing with KGE loss and spatial regularization produces **identical** optimization trajectories (same loss values, same gradients to machine precision) as the current DODO `CoupledFUSERoute` on the Bow-at-Banff test case.

### 12.2 Three-Component Test
Adding dGW as a third component (FUSE → dGW → dRoute) works with:
- Gradients flowing through all three components
- Conservation checker verifying water balance
- Joint optimization of FUSE params + GW params + Manning's n

### 12.3 Multi-Gauge Test
Optimizing against 5+ streamflow gauges simultaneously, with a single coupling graph, produces better spatially distributed parameters than single-gauge optimization.

### 12.4 Standalone Simplicity Test
A new user can run the following in under 50 lines:

```python
import dcoupler as dc

# Create components
land = dc.components.SimpleBucket(n_hrus=10, dt=86400)
router = dc.components.LagRouter(n_reaches=10, dt=86400)

# Wire them
graph = dc.CouplingGraph()
graph.add_component(land)
graph.add_component(router)
graph.connect("land", "runoff", "router", "lateral_inflow",
              spatial_remap=dc.SpatialRemapper.identity(10))

# Observe
obs_operator = dc.observers.StreamflowObserver(gauge_reach_ids=[9])
loss = dc.MultiObservationLoss()
loss.add_term(obs_operator, observed_Q, loss_fn="nse")

# Train
trainer = dc.Trainer(graph, loss, lr=0.01, n_epochs=100)
result = trainer.train(forcing_data, n_timesteps=365, dt=86400)
```

---

## 13. Dependencies

### Required
- Python >= 3.10
- PyTorch >= 2.0
- numpy
- scipy (sparse matrices for remapping)

### Optional
- xarray (for NetCDF I/O helpers)
- jax + jax2torch (for JAX component wrappers)

### Not Dependencies (provided by SYMFLUENCE or user)
- Specific model codes (cfuse, droute, etc.)
- Data acquisition tools
- HPC infrastructure
- Visualization

---

## 14. Relationship to SYMFLUENCE

SYMFLUENCE interacts with dCoupler at two levels:

### 14.1 Configuration Generation
SYMFLUENCE reads its own config files and generates:
- Component instances (with paths to model executables, parameter files, etc.)
- Spatial remapping matrices (from GIS preprocessing)
- Forcing data tensors (from data acquisition pipeline)
- Observation data tensors (from observation databases)

### 14.2 Job Orchestration
SYMFLUENCE launches dCoupler training runs:
- Sets up GPU/HPC environment
- Manages checkpoints across job restarts
- Archives results (trained parameters, diagnostics, training history)
- Runs forward prediction with trained parameters

### 14.3 Interface Contract
```python
# SYMFLUENCE produces:
dcoupler_config = {
    "components": [...],           # Instantiated DifferentiableComponent objects
    "connections": [...],           # FluxConnection specs
    "observations": [...],          # ObservationOperator + observed data
    "forcing": {...},               # External input tensors
    "optimization": {               # Trainer config
        "lr": 0.01,
        "n_epochs": 500,
        "scheduler": "warm_restarts",
        ...
    }
}

# dCoupler consumes it and produces:
result = dcoupler.run(dcoupler_config)
# result contains: trained parameters, training history, diagnostics
# SYMFLUENCE archives result
```

---

## 15. Implementation Priority

### Phase 1: Core (Week 1-2)
- `DifferentiableComponent` protocol
- `FluxConnection` with `SpatialRemapper`
- `CouplingGraph` with topological sort and basic `forward()`
- `ParameterManager` with bounded transforms
- Losses migrated from DODO

### Phase 2: First Components (Week 3-4)
- `EnzymeComponent` wrapper (extract pattern from DODO)
- FUSE component wrapping `DifferentiableFUSEBatch`
- Muskingum-Cunge routing component wrapping `DifferentiableRouting`
- **Equivalence test**: reproduce DODO results exactly

### Phase 3: Training Infrastructure (Week 5-6)
- `Trainer` with full training loop
- `GradientDiagnostics`
- Multi-observation loss
- Checkpoint save/restore
- **Standalone simplicity test**

### Phase 4: Temporal Orchestration (Week 7-8)
- `TemporalOrchestrator` with operator splitting
- Components running at different timesteps
- Temporal interpolation of fluxes
- `ConservationChecker`

### Phase 5: Third Component (Week 9-10)
- dGW component wrapper
- FUSE → dGW → dRoute three-way coupling
- **Three-component test**
- **Multi-gauge test**

### Phase 6: Observation Operators (Week 11-12)
- `StreamflowObserver` (trivial, but formalized)
- `GRACEObserver`
- `SoilMoistureObserver`
- Multi-observation calibration demonstration

---

## 16. Open Design Questions

1. **PyTorch vs JAX for the coupler itself?** DODO uses PyTorch. jSUMMA/dLand use JAX. The coupler needs to bridge both. Current recommendation: PyTorch as the coupling layer (mature autograd, broader ecosystem), with jax2torch bridges for JAX components. Revisit if JAX components dominate.

2. **Implicit vs explicit time stepping for bidirectional coupling?** When Component A and Component B exchange fluxes bidirectionally within a timestep, do we iterate to convergence (implicit) or use previous-timestep values (explicit/lagged)? Implicit is more accurate but requires fixed-point iteration within the computational graph. Start with explicit, add implicit later.

3. **GPU support?** DODO runs on CPU. For large-domain training with many HRUs, GPU acceleration of the coupling graph forward pass would be valuable. The `SpatialRemapper` sparse matmul and loss computations are straightforwardly GPU-compatible. Component `step()` calls depend on the component backend. Design for CPU-first, GPU-optional.

4. **Distributed training?** For continent-scale applications, the coupling graph may need to be partitioned across multiple GPUs/nodes. This is a future concern but the `CouplingGraph` API should not preclude it.

---

*End of specification.*
