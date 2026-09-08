from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from experiments.v837_primitive_invention.v837y.candidate_interaction import (
    CandidateInteractionSpec,
    ControlledCandidateInteractionModel,
)

INPUT_DIM = 6
NUM_CELLS = 10
CONDITIONS = {
    "AF0_y3_parent",
    "AF1_shared_candidate_input_factorization",
    "AF1F_folded_candidate_input_control",
    "AF1D_deshared_candidate_input_factorization",
}


@dataclass
class CandidateInputProjectionSnapshot:
    singular_values: list[float]
    condition_number: float | None
    frobenius_norm: float
    bias_norm: float
    weight_drift: float
    bias_drift: float


class CandidateInputFactorizationY3(ControlledCandidateInteractionModel):
    """Exact Y3 with only the candidate-input parameterization varied.

    AF1/AF1D are initialized to the exact historical Y3 effective candidate
    input maps. The projection therefore changes optimization geometry without
    changing input visibility or the step-zero function. AF1F folds that same
    initialization back into each cell's direct input map and executes no
    projection at runtime.
    """

    architecture_name = "v837af_candidate_input_factorization_y3"

    def __init__(
        self,
        graph,
        *,
        condition: str,
        coupling_initialization_seed: int,
        projection_seed: int,
    ):
        if condition not in CONDITIONS:
            raise ValueError(f"unknown V837af condition: {condition}")
        self.transfer_condition = condition
        spec = CandidateInteractionSpec(
            global_scalar_control=True,
            candidate_coupling_mode="rank4_cross_block",
            coupling_rank=4,
        )
        super().__init__(
            graph,
            spec=spec,
            coupling_initialization_seed=int(coupling_initialization_seed),
            obs_dim=INPUT_DIM,
        )

        # Historical high-capacity/Y3 graph uses broadcast input. Keep generic
        # support for masked access by projecting only _visible_input output.
        # Freeze a parent-preserving deep-linear split. Identity/zero makes
        # AF1, AF1F and AF1D exactly the same historical Y3 function at step
        # zero while still exposing the full trainable 6x6+bias factorization.
        self.projection_initialization_seed = int(projection_seed)
        A = torch.eye(INPUT_DIM, dtype=self.base.cell_wx[0].dtype)
        a = torch.zeros(INPUT_DIM, dtype=self.base.cell_wx[0].dtype)
        self.register_buffer("diagnostic_projection_weight", A.clone(), persistent=False)
        self.register_buffer("diagnostic_projection_bias", a.clone(), persistent=False)

        original_w = [p.detach().clone() for p in self.base.cell_wx]
        original_b = [p.detach().clone() for p in self.base.cell_b]
        downstream_w: list[torch.Tensor] = []
        downstream_b: list[torch.Tensor] = []
        for W, b in zip(original_w, original_b):
            # Q A = W, d + Q a = b.
            Q = torch.linalg.solve(A.T, W.T).T
            d = b - Q @ a
            downstream_w.append(Q)
            downstream_b.append(d)

        self.shared_candidate_projection: nn.Linear | None = None
        self.cell_candidate_projections: nn.ModuleList | None = None

        if condition == "AF1_shared_candidate_input_factorization":
            layer = nn.Linear(INPUT_DIM, INPUT_DIM)
            with torch.no_grad():
                layer.weight.copy_(A)
                layer.bias.copy_(a)
                for i in range(NUM_CELLS):
                    self.base.cell_wx[i].copy_(downstream_w[i])
                    self.base.cell_b[i].copy_(downstream_b[i])
            self.shared_candidate_projection = layer
            self.register_buffer("initial_shared_projection_weight", A.clone(), persistent=False)
            self.register_buffer("initial_shared_projection_bias", a.clone(), persistent=False)
        elif condition == "AF1D_deshared_candidate_input_factorization":
            layers = nn.ModuleList()
            with torch.no_grad():
                for i in range(NUM_CELLS):
                    layer = nn.Linear(INPUT_DIM, INPUT_DIM)
                    layer.weight.copy_(A)
                    layer.bias.copy_(a)
                    layers.append(layer)
                    self.base.cell_wx[i].copy_(downstream_w[i])
                    self.base.cell_b[i].copy_(downstream_b[i])
            self.cell_candidate_projections = layers
            self.register_buffer("initial_shared_projection_weight", A.clone(), persistent=False)
            self.register_buffer("initial_shared_projection_bias", a.clone(), persistent=False)
        elif condition == "AF1F_folded_candidate_input_control":
            # Derive AF1 first, then algebraically fold it. No runtime projection.
            with torch.no_grad():
                for i in range(NUM_CELLS):
                    folded_w = downstream_w[i] @ A
                    folded_b = downstream_b[i] + downstream_w[i] @ a
                    self.base.cell_wx[i].copy_(folded_w)
                    self.base.cell_b[i].copy_(folded_b)
            self.register_buffer("initial_shared_projection_weight", A.clone(), persistent=False)
            self.register_buffer("initial_shared_projection_bias", a.clone(), persistent=False)
        else:
            self.register_buffer("initial_shared_projection_weight", torch.empty(0), persistent=False)
            self.register_buffer("initial_shared_projection_bias", torch.empty(0), persistent=False)
        self._shared_projection_cache_key = None
        self._shared_projection_cache_value = None

    @property
    def projection_mode(self) -> str:
        if self.transfer_condition == "AF1_shared_candidate_input_factorization":
            return "shared"
        if self.transfer_condition == "AF1D_deshared_candidate_input_factorization":
            return "deshared"
        if self.transfer_condition == "AF1F_folded_candidate_input_control":
            return "folded"
        return "none"

    @property
    def runtime_projection_active(self) -> bool:
        return self.projection_mode in {"shared", "deshared"}

    @property
    def projection_specific_macs(self) -> int:
        if self.projection_mode == "shared":
            return 36 if self.base.input_access_mode in {"broadcast", "none"} else 360
        if self.projection_mode == "deshared":
            return 360
        return 0

    @property
    def total_recurrent_controller_projection_macs(self) -> int:
        return int(self.recurrent_controller_macs + self.projection_specific_macs)

    @property
    def projection_parameter_count(self) -> int:
        if self.projection_mode == "shared":
            return 42
        if self.projection_mode == "deshared":
            return 420
        return 0

    def diagnostic_projected_visible_input(self, visible_x: torch.Tensor, cell_index: int) -> torch.Tensor:
        """Projection intermediate for paired step-zero diagnostics.

        AF1F computes this only when explicitly called for diagnostics; its
        runtime forward never calls a projection.
        """
        if self.projection_mode == "shared":
            assert self.shared_candidate_projection is not None
            return self.shared_candidate_projection(visible_x)
        if self.projection_mode == "deshared":
            assert self.cell_candidate_projections is not None
            return self.cell_candidate_projections[cell_index](visible_x)
        if self.projection_mode == "folded":
            return F.linear(visible_x, self.diagnostic_projection_weight, self.diagnostic_projection_bias)
        return visible_x

    def forward(self, *args, **kwargs):
        self._shared_projection_cache_key = None
        self._shared_projection_cache_value = None
        return super().forward(*args, **kwargs)

    def _visible_input(self, x_t: torch.Tensor, cell_index: int) -> torch.Tensor:
        visible = super()._visible_input(x_t, cell_index)
        if self.projection_mode == "shared":
            assert self.shared_candidate_projection is not None
            if self.base.input_access_mode in {"broadcast", "none"}:
                key = (int(x_t.data_ptr()), int(x_t.storage_offset()), tuple(x_t.shape))
                if self._shared_projection_cache_key != key:
                    self._shared_projection_cache_key = key
                    self._shared_projection_cache_value = self.shared_candidate_projection(visible)
                assert self._shared_projection_cache_value is not None
                return self._shared_projection_cache_value
            return self.shared_candidate_projection(visible)
        if self.projection_mode == "deshared":
            assert self.cell_candidate_projections is not None
            return self.cell_candidate_projections[cell_index](visible)
        return visible

    def effective_candidate_input_map(self, cell_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        W = self.base.cell_wx[cell_index]
        b = self.base.cell_b[cell_index]
        if self.projection_mode == "shared":
            assert self.shared_candidate_projection is not None
            A = self.shared_candidate_projection.weight
            a = self.shared_candidate_projection.bias
            return W @ A, b + W @ a
        if self.projection_mode == "deshared":
            assert self.cell_candidate_projections is not None
            layer = self.cell_candidate_projections[cell_index]
            return W @ layer.weight, b + W @ layer.bias
        return W, b

    @staticmethod
    def _snapshot(weight: torch.Tensor, bias: torch.Tensor, iw: torch.Tensor, ib: torch.Tensor) -> CandidateInputProjectionSnapshot:
        with torch.no_grad():
            w = weight.detach().cpu()
            b = bias.detach().cpu()
            singular = torch.linalg.svdvals(w)
            minimum = float(singular.min().item()) if singular.numel() else 0.0
            maximum = float(singular.max().item()) if singular.numel() else 0.0
            return CandidateInputProjectionSnapshot(
                singular_values=[float(v) for v in singular.tolist()],
                condition_number=float(maximum / max(minimum, 1e-12)) if singular.numel() else None,
                frobenius_norm=float(torch.linalg.vector_norm(w).item()),
                bias_norm=float(torch.linalg.vector_norm(b).item()),
                weight_drift=float(torch.linalg.vector_norm(w - iw.detach().cpu()).item()),
                bias_drift=float(torch.linalg.vector_norm(b - ib.detach().cpu()).item()),
            )

    def projection_diagnostics(self) -> dict:
        if self.projection_mode == "shared":
            assert self.shared_candidate_projection is not None
            s = self._snapshot(
                self.shared_candidate_projection.weight,
                self.shared_candidate_projection.bias,
                self.initial_shared_projection_weight,
                self.initial_shared_projection_bias,
            )
            return {"mode": "shared", "runtime_active": True, **s.__dict__}
        if self.projection_mode == "deshared":
            assert self.cell_candidate_projections is not None
            rows = []
            for layer in self.cell_candidate_projections:
                rows.append(self._snapshot(
                    layer.weight, layer.bias,
                    self.initial_shared_projection_weight,
                    self.initial_shared_projection_bias,
                ).__dict__)
            flat = torch.stack([layer.weight.detach().reshape(-1) for layer in self.cell_candidate_projections])
            distances = []
            cosines = []
            for i in range(NUM_CELLS):
                for j in range(i + 1, NUM_CELLS):
                    distances.append(float(torch.linalg.vector_norm(flat[i] - flat[j]).item()))
                    den = max(float(torch.linalg.vector_norm(flat[i]) * torch.linalg.vector_norm(flat[j])), 1e-12)
                    cosines.append(float(torch.dot(flat[i], flat[j]).item() / den))
            return {
                "mode": "deshared",
                "runtime_active": True,
                "per_cell": rows,
                "pairwise_weight_distance_median": float(torch.tensor(distances).median().item()) if distances else 0.0,
                "pairwise_weight_cosine_median": float(torch.tensor(cosines).median().item()) if cosines else 1.0,
                "projection_divergence_from_shared_initialization_mean": float(sum(r["weight_drift"] for r in rows) / len(rows)),
            }
        return {
            "mode": self.projection_mode,
            "runtime_active": False,
            "singular_values": [],
            "condition_number": None,
            "frobenius_norm": 0.0,
            "bias_norm": 0.0,
            "weight_drift": 0.0,
            "bias_drift": 0.0,
        }

    def effective_input_map_diagnostics(self) -> dict:
        with torch.no_grad():
            maps = []
            rows = []
            for i in range(NUM_CELLS):
                W, b = self.effective_candidate_input_map(i)
                flat = W.detach().reshape(-1)
                maps.append(flat)
                rows.append({
                    "cell": i,
                    "weight": W.detach().cpu().tolist(),
                    "bias": b.detach().cpu().tolist(),
                    "weight_norm": float(torch.linalg.vector_norm(W.detach()).item()),
                    "bias_norm": float(torch.linalg.vector_norm(b.detach()).item()),
                })
            cosines = []
            distances = []
            for i in range(NUM_CELLS):
                for j in range(i + 1, NUM_CELLS):
                    distances.append(float(torch.linalg.vector_norm(maps[i] - maps[j]).item()))
                    den = max(float(torch.linalg.vector_norm(maps[i]) * torch.linalg.vector_norm(maps[j])), 1e-12)
                    cosines.append(float(torch.dot(maps[i], maps[j]).item() / den))
            return {
                "per_cell": rows,
                "pairwise_cosine_median": float(torch.tensor(cosines).median().item()) if cosines else 1.0,
                "pairwise_distance_median": float(torch.tensor(distances).median().item()) if distances else 0.0,
            }
