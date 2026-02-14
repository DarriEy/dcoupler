from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch


@dataclass
class FluxConnection:
    """
    A differentiable connection between two component flux ports.
    """

    source_component: str
    source_flux: str
    target_component: str
    target_flux: str
    spatial_remap: Optional["SpatialRemapper"] = None
    unit_conversion: Optional[float] = None
    temporal_interp: Optional[str] = None
    conserved_quantity: Optional[str] = None


class SpatialRemapper:
    """
    Differentiable spatial remapping between component grids.

    All remapping methods produce a sparse matrix M such that:
        target_flux = M @ source_flux
    """

    def __init__(self, matrix: torch.Tensor):
        if not matrix.is_sparse:
            raise ValueError("SpatialRemapper expects a sparse torch tensor")
        self.matrix = matrix.coalesce()

    @staticmethod
    def from_sparse(matrix: torch.Tensor) -> "SpatialRemapper":
        return SpatialRemapper(matrix)

    @staticmethod
    def from_mapping_table(
        source_ids: np.ndarray,
        target_ids: np.ndarray,
        hru_to_seg: np.ndarray,
        areas: Optional[np.ndarray] = None,
        unit_factor: float = 1.0,
    ) -> "SpatialRemapper":
        """
        Build remapper from explicit ID mapping.

        Args:
            source_ids: IDs in source order (length n_source)
            target_ids: IDs in target order (length n_target)
            hru_to_seg: Mapping from each source index -> target ID
            areas: Optional per-source area weights
            unit_factor: Multiplicative factor applied to each mapping weight
        """
        source_ids = np.asarray(source_ids)
        target_ids = np.asarray(target_ids)
        hru_to_seg = np.asarray(hru_to_seg)

        id_to_target = {int(tid): i for i, tid in enumerate(target_ids)}
        rows = []
        cols = []
        vals = []

        for src_idx, target_id in enumerate(hru_to_seg):
            target_idx = id_to_target.get(int(target_id))
            if target_idx is None:
                continue
            weight = float(unit_factor)
            if areas is not None:
                weight *= float(areas[src_idx])
            rows.append(target_idx)
            cols.append(src_idx)
            vals.append(weight)

        if len(rows) == 0:
            raise ValueError("No valid mappings found for remapper")

        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        matrix = torch.sparse_coo_tensor(
            indices,
            values,
            (len(target_ids), len(source_ids)),
        )
        return SpatialRemapper(matrix)

    @staticmethod
    def from_grid_overlap(
        source_grid: "GridSpec",
        target_grid: "GridSpec",
        method: str = "conservative",
    ) -> "SpatialRemapper":
        raise NotImplementedError(
            "GridSpec-based remapping is not implemented. "
            "Provide a precomputed sparse matrix via SpatialRemapper.from_sparse."
        )

    @staticmethod
    def identity(n: int) -> "SpatialRemapper":
        indices = torch.arange(n, dtype=torch.long)
        index = torch.stack([indices, indices], dim=0)
        values = torch.ones(n, dtype=torch.float32)
        matrix = torch.sparse_coo_tensor(index, values, (n, n))
        return SpatialRemapper(matrix)

    def forward(self, flux: torch.Tensor) -> torch.Tensor:
        """Apply remapping. Differentiable via sparse matmul."""
        if flux.dim() == 1:
            return torch.sparse.mm(self.matrix, flux.unsqueeze(1)).squeeze(1)
        if flux.dim() == 2:
            # flux: [time, source] -> [time, target]
            return torch.sparse.mm(self.matrix, flux.T).T
        raise ValueError("Flux tensor must be 1D or 2D")
