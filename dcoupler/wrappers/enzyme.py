from __future__ import annotations

import torch
import numpy as np

from dcoupler.core.component import DifferentiableComponent, GradientMethod


class EnzymeComponent(DifferentiableComponent):
    """Base class for C++ models with Enzyme AD."""

    gradient_method = GradientMethod.ENZYME


class DifferentiableRouting(torch.autograd.Function):
    """
    PyTorch wrapper for dRoute using Enzyme AD gradients.

    Forward: Standard MuskingumCungeRouter
    Backward: Enzyme AD via dmc.enzyme.compute_manning_gradients
    """

    @staticmethod
    def forward(
        ctx,
        lateral_inflows: torch.Tensor,  # [n_timesteps, n_reaches]
        manning_n: torch.Tensor,  # [n_reaches]
        router,
        network,
        outlet_reach_id: int,
        dt: float,
    ) -> torch.Tensor:
        ctx.router = router
        ctx.network = network
        ctx.outlet_reach_id = outlet_reach_id
        ctx.n_reaches = lateral_inflows.shape[1]
        ctx.n_timesteps = lateral_inflows.shape[0]
        ctx.dt = dt

        inflows_np = lateral_inflows.detach().cpu().numpy().astype(np.float64)
        manning_np = manning_n.detach().cpu().numpy().astype(np.float64)

        network.set_manning_n_all(manning_np)
        router.reset_state()

        topo_order_ids = list(network.topological_order())
        outlet_Q = []
        for t in range(ctx.n_timesteps):
            for i, rid in enumerate(topo_order_ids):
                router.set_lateral_inflow(int(rid), float(inflows_np[t, i]))
            router.route_timestep()
            outlet_Q.append(router.get_discharge(outlet_reach_id))

        outlet_Q = np.array(outlet_Q, dtype=np.float64)

        n_reaches = ctx.n_reaches
        topo_order = np.arange(n_reaches, dtype=np.int32)
        id_to_idx = {int(rid): i for i, rid in enumerate(topo_order_ids)}

        upstream_counts = np.zeros(n_reaches, dtype=np.int32)
        upstream_lists = [[] for _ in range(n_reaches)]
        for i, rid in enumerate(topo_order_ids):
            reach = network.get_reach(int(rid))
            if reach.upstream_junction_id >= 0:
                try:
                    junc = network.get_junction(reach.upstream_junction_id)
                    for up_id in junc.upstream_reach_ids:
                        if up_id in id_to_idx:
                            upstream_lists[i].append(id_to_idx[up_id])
                except Exception:
                    pass

        for i in range(n_reaches):
            upstream_counts[i] = len(upstream_lists[i])

        upstream_offsets = np.zeros(n_reaches + 1, dtype=np.int32)
        for i in range(n_reaches):
            upstream_offsets[i + 1] = upstream_offsets[i] + upstream_counts[i]

        total_upstream = upstream_offsets[n_reaches]
        upstream_indices = np.zeros(max(total_upstream, 1), dtype=np.int32)
        for i in range(n_reaches):
            offset = upstream_offsets[i]
            for j, up_idx in enumerate(upstream_lists[i]):
                upstream_indices[offset + j] = up_idx

        lengths = np.zeros(n_reaches, dtype=np.float64)
        slopes = np.zeros(n_reaches, dtype=np.float64)
        width_coefs = np.zeros(n_reaches, dtype=np.float64)
        width_exps = np.zeros(n_reaches, dtype=np.float64)
        depth_coefs = np.zeros(n_reaches, dtype=np.float64)
        depth_exps = np.zeros(n_reaches, dtype=np.float64)

        for i, rid in enumerate(topo_order_ids):
            reach = network.get_reach(int(rid))
            lengths[i] = reach.length
            slopes[i] = max(reach.slope, 0.0001)
            width_coefs[i] = reach.geometry.width_coef
            width_exps[i] = reach.geometry.width_exp
            depth_coefs[i] = reach.geometry.depth_coef
            depth_exps[i] = reach.geometry.depth_exp

        outlet_idx = id_to_idx[outlet_reach_id]

        ctx.save_for_backward(lateral_inflows, manning_n)
        ctx.inflows_np = inflows_np
        ctx.topo_order = topo_order
        ctx.topo_order_ids = topo_order_ids
        ctx.upstream_counts = upstream_counts
        ctx.upstream_offsets = upstream_offsets
        ctx.upstream_indices = upstream_indices
        ctx.lengths = lengths
        ctx.slopes = slopes
        ctx.width_coefs = width_coefs
        ctx.width_exps = width_exps
        ctx.depth_coefs = depth_coefs
        ctx.depth_exps = depth_exps
        ctx.outlet_idx = outlet_idx

        return torch.tensor(outlet_Q, dtype=torch.float32, device=lateral_inflows.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        import droute as dmc

        lateral_inflows, manning_n = ctx.saved_tensors

        grad_np = grad_output.detach().cpu().numpy().astype(np.float64)
        manning_np = manning_n.detach().cpu().numpy().astype(np.float64)

        try:
            grad_manning_np = dmc.enzyme.compute_manning_gradients(
                manning_np,
                ctx.inflows_np,
                grad_np,
                ctx.lengths,
                ctx.slopes,
                ctx.width_coefs,
                ctx.width_exps,
                ctx.depth_coefs,
                ctx.depth_exps,
                ctx.topo_order,
                ctx.upstream_counts,
                ctx.upstream_offsets,
                ctx.upstream_indices,
                ctx.outlet_idx,
                ctx.dt,
            )
            grad_manning_np = np.array(grad_manning_np)
        except Exception:
            eps = 0.01
            grad_manning_np = np.zeros(ctx.n_reaches)
            router = ctx.router
            network = ctx.network
            topo_order_ids = ctx.topo_order_ids

            for i in range(ctx.n_reaches):
                mann_pert = manning_np.copy()
                mann_pert[i] = manning_np[i] * (1 + eps)
                network.set_manning_n_all(mann_pert)
                router.reset_state()
                Q_plus = []
                for t in range(ctx.n_timesteps):
                    for j, rid in enumerate(topo_order_ids):
                        router.set_lateral_inflow(int(rid), float(ctx.inflows_np[t, j]))
                    router.route_timestep()
                    Q_plus.append(router.get_discharge(ctx.outlet_reach_id))

                mann_pert[i] = manning_np[i] * (1 - eps)
                network.set_manning_n_all(mann_pert)
                router.reset_state()
                Q_minus = []
                for t in range(ctx.n_timesteps):
                    for j, rid in enumerate(topo_order_ids):
                        router.set_lateral_inflow(int(rid), float(ctx.inflows_np[t, j]))
                    router.route_timestep()
                    Q_minus.append(router.get_discharge(ctx.outlet_reach_id))

                dQ_dn = (np.array(Q_plus) - np.array(Q_minus)) / (2 * eps * manning_np[i])
                grad_manning_np[i] = np.sum(grad_np * dQ_dn)

            network.set_manning_n_all(manning_np)

        grad_manning = torch.from_numpy(grad_manning_np.astype(np.float32))
        grad_lateral = grad_np[:, np.newaxis] * np.ones((1, ctx.n_reaches))
        grad_lateral_t = torch.from_numpy(grad_lateral.astype(np.float32))

        return grad_lateral_t, grad_manning, None, None, None, None
