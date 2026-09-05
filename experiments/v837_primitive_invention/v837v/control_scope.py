from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.v837_primitive_invention.common.substrate import ForwardTrace
from experiments.v837_primitive_invention.v837u.dynamic_control import NeutralDynamicCarryModel


@dataclass(frozen=True)
class ControlDomainSpec:
    name: str
    num_cells: int
    domains: tuple[tuple[int, ...], ...]
    source_cells: tuple[int, ...]

    def validate(self) -> None:
        if len(self.domains) != len(self.source_cells):
            raise ValueError("every control domain requires exactly one source")
        flat = [cell for domain in self.domains for cell in domain]
        if sorted(flat) != list(range(self.num_cells)):
            raise ValueError("every cell must appear in exactly one control domain")
        for domain, source in zip(self.domains, self.source_cells):
            if source not in domain:
                raise ValueError("control-domain source must belong to its domain")
            if source != min(domain):
                raise ValueError("control-domain source must be the earliest cell in its domain")
            if tuple(sorted(domain)) != domain:
                raise ValueError("control domains must preserve deterministic execution order")

    @property
    def domain_count(self) -> int:
        return len(self.domains)

    def source_for_cell(self, cell: int) -> int:
        for domain, source in zip(self.domains, self.source_cells):
            if cell in domain:
                return source
        raise KeyError(cell)


DOMAIN_SPECS = {
    "V0_10_domains": ControlDomainSpec(
        name="V0_10_domains", num_cells=10,
        domains=((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)),
        source_cells=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    ),
    "V1_5_domains": ControlDomainSpec(
        name="V1_5_domains", num_cells=10,
        domains=((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)),
        source_cells=(0, 2, 4, 6, 8),
    ),
    "V2_2_domains": ControlDomainSpec(
        name="V2_2_domains", num_cells=10,
        domains=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)),
        source_cells=(0, 5),
    ),
    "V3_1_domain": ControlDomainSpec(
        name="V3_1_domain", num_cells=10,
        domains=((0, 1, 2, 3, 4, 5, 6, 7, 8, 9),),
        source_cells=(0,),
    ),
}
for _spec in DOMAIN_SPECS.values():
    _spec.validate()


class NeutralControlScopeModel(NeutralDynamicCarryModel):
    """V837v: preserve local controller information while varying output scope only.

    Each domain uses the gate produced by one fixed existing local source cell.
    Followers do not compute or consume their own controller outputs in the primary
    forward path. No pooling, global observer, extra state visibility, or new
    message visibility is introduced.
    """

    def __init__(self, graph, *, domain_spec: ControlDomainSpec, obs_dim: int = 6):
        if len(graph.cells) != domain_spec.num_cells:
            raise ValueError("control-domain layout must match graph cell count")
        domain_spec.validate()
        self.control_domain_spec = domain_spec
        super().__init__(graph, condition="U2_dynamic_scalar_carry", obs_dim=obs_dim)

    @property
    def nominal_controller_count(self) -> int:
        return 10

    @property
    def active_controller_count(self) -> int:
        return self.control_domain_spec.domain_count

    @property
    def nominal_controller_parameter_count(self) -> int:
        return 150

    @property
    def active_controller_parameter_count(self) -> int:
        return 15 * self.active_controller_count

    @property
    def controller_macs_per_timestep(self) -> int:
        return 14 * self.active_controller_count

    @property
    def total_recurrent_controller_macs_per_timestep(self) -> int:
        return 160 + self.controller_macs_per_timestep

    def _visible_input(self, x_t: torch.Tensor, cell: int) -> torch.Tensor:
        if self.input_access_mode == "none":
            return torch.zeros_like(x_t)
        if self.input_access_mode == "broadcast":
            return x_t
        return x_t * self.input_access_mask[:, cell].view(1, -1)

    def forward(
        self,
        observations: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        return_trace: bool = False,
        domain_spec_override: ControlDomainSpec | None = None,
        disable_messages: bool = False,
    ):
        if observations.ndim != 3:
            raise ValueError("observations must be [B,T,D]")
        batch, steps, observed_dim = observations.shape
        if observed_dim != self.obs_dim:
            raise ValueError(f"observation dimension {observed_dim} != model obs_dim {self.obs_dim}")
        spec = domain_spec_override or self.control_domain_spec
        spec.validate()
        if spec.num_cells != len(self.graph.cells):
            raise ValueError("override domain layout must match graph")

        n = len(self.graph.cells)
        device = observations.device
        prev_states = [torch.zeros(batch, self.state_dim, device=device) for _ in range(n)]
        prev_outputs = [torch.zeros(batch, self.message_dim, device=device) for _ in range(n)]
        st, cand, outs, msgs, recs, mterms, iterms, mods = [], [], [], [], [], [], [], []

        for t in range(steps):
            x_t = observations[:, t, :]
            cs, cc, co, cm, cr, cmt, cit, cg = [], [], [], [], [], [], [], []
            source_gates: dict[int, torch.Tensor] = {}
            for i in range(n):
                message = torch.zeros(batch, self.message_dim, device=device)
                if not disable_messages:
                    for edge_index, edge in enumerate(self.graph.edges):
                        if edge.dst != i:
                            continue
                        source = prev_outputs[edge.src] if edge.recurrent or edge.src >= len(co) else co[edge.src]
                        message = message + self.edge_weights[edge_index] * source
                visible_x = self._visible_input(x_t, i)
                source = spec.source_for_cell(i)
                if i == source:
                    g = torch.sigmoid(
                        torch.sum(prev_states[i] * self.cell_gs[i].view(1, -1), dim=1, keepdim=True)
                        + torch.sum(message * self.cell_gm[i].view(1, -1), dim=1, keepdim=True)
                        + torch.sum(visible_x * self.cell_gx[i].view(1, -1), dim=1, keepdim=True)
                        + self.cell_gb[i]
                    )
                    source_gates[i] = g
                else:
                    if source not in source_gates:
                        raise RuntimeError("domain source gate unavailable before follower update")
                    g = source_gates[source]

                recurrent_term = prev_states[i] @ self.cell_ws[i].T
                message_term = message @ self.cell_wm[i].T
                input_term = visible_x @ self.cell_wx[i].T
                candidate = torch.tanh(recurrent_term + message_term + input_term + self.cell_b[i])
                proposed = g * prev_states[i] + (1.0 - g) * candidate
                proposed_output = proposed @ self.cell_wo[i].T

                if lengths is not None:
                    active = (t < lengths).to(observations.dtype).unsqueeze(1)
                    state = active * proposed + (1.0 - active) * prev_states[i]
                    output = active * proposed_output + (1.0 - active) * prev_outputs[i]
                    if return_trace:
                        message = active * message
                        recurrent_term = active * recurrent_term
                        message_term = active * message_term
                        input_term = active * input_term
                        g = active * g + (1.0 - active)
                else:
                    state, output = proposed, proposed_output

                cs.append(state)
                cc.append(candidate)
                co.append(output)
                if return_trace:
                    cm.append(message)
                    cr.append(recurrent_term)
                    cmt.append(message_term)
                    cit.append(input_term)
                    cg.append(g)

            prev_states, prev_outputs = cs, co
            if return_trace:
                st.append(torch.stack(cs, 1))
                cand.append(torch.stack(cc, 1))
                outs.append(torch.stack(co, 1))
                msgs.append(torch.stack(cm, 1))
                recs.append(torch.stack(cr, 1))
                mterms.append(torch.stack(cmt, 1))
                iterms.append(torch.stack(cit, 1))
                mods.append(torch.stack(cg, 1))

        prediction = torch.tanh(self.readout(torch.cat(prev_states, dim=1))).squeeze(-1)
        if not return_trace:
            return prediction
        return prediction, ForwardTrace(
            states=torch.stack(st, 1),
            candidate_states=torch.stack(cand, 1),
            outputs=torch.stack(outs, 1),
            messages=torch.stack(msgs, 1),
            recurrent_terms=torch.stack(recs, 1),
            message_terms=torch.stack(mterms, 1),
            input_terms=torch.stack(iterms, 1),
            state_modulators=torch.stack(mods, 1),
        )
