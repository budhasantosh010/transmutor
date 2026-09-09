from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837ak.reconstruct_organisms import load_reconstructed_model
from experiments.v837_primitive_invention.v837ak.utils import HERE, ROOT, read_json

CACHE = HERE / "raw" / "cache" / "traces"


@dataclass
class FullProbeTrace:
    observations: torch.Tensor
    lengths: torch.Tensor
    targets: torch.Tensor
    states: torch.Tensor
    candidates: torch.Tensor
    outputs: torch.Tensor
    messages: torch.Tensor
    local_terms: torch.Tensor
    global_terms: torch.Tensor
    matched_terms: torch.Tensor
    message_terms: torch.Tensor
    input_terms: torch.Tensor
    gates: torch.Tensor


def _reconstruction_row(organism_id: str) -> dict:
    payload = read_json(HERE / "raw/reconstruction_results.json")
    return next(row for row in payload["rows"] if row["organism_id"] == organism_id)


def probe_batch(row: dict, seeds: Sequence[int], split: str = "validation"):
    task = task_by_name(row["family"])
    episodes = [task.generate(int(seed), split) for seed in seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    return task, episodes, observations, lengths, targets


def run_full_probe(organism_id: str, seeds: Sequence[int], *, cache_key: str | None = None) -> FullProbeTrace:
    row = _reconstruction_row(organism_id)
    cache_path = CACHE / f"{organism_id}__{cache_key}.pt" if cache_key else None
    if cache_path is not None and cache_path.is_file():
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        if data.get("seeds") == [int(s) for s in seeds] and data.get("final_state_hash") == row["final_state_hash"]:
            return FullProbeTrace(**data["trace"])
    model, _, checkpoint = load_reconstructed_model(row)
    _, _, observations, lengths, targets = probe_batch(row, seeds)
    model.eval()
    with torch.no_grad():
        _, trace = model(observations, lengths, return_trace=True)
    payload = FullProbeTrace(
        observations=observations.detach().cpu(), lengths=lengths.detach().cpu(), targets=targets.detach().cpu(),
        states=trace.states.detach().cpu(), candidates=trace.candidate_states.detach().cpu(), outputs=trace.outputs.detach().cpu(),
        messages=trace.messages.detach().cpu(), local_terms=trace.recurrent_terms.detach().cpu(),
        global_terms=trace.global_recurrent_terms.detach().cpu(), matched_terms=trace.matched_local_terms.detach().cpu(),
        message_terms=trace.message_terms.detach().cpu(), input_terms=trace.input_terms.detach().cpu(),
        gates=trace.global_gates.detach().cpu(),
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version":"V837ak","organism_id":organism_id,"seeds":[int(s) for s in seeds],"final_state_hash":checkpoint["final_state_hash"],"trace":payload.__dict__}, cache_path)
    return payload


def active_mask(trace: FullProbeTrace) -> torch.Tensor:
    t = torch.arange(trace.states.shape[1]).view(1, -1)
    return t < trace.lengths.view(-1, 1)


def previous_outputs(outputs: torch.Tensor) -> torch.Tensor:
    prev = torch.zeros_like(outputs)
    prev[:, 1:] = outputs[:, :-1]
    return prev


def internal_raw_messages(model, trace: FullProbeTrace, nodes: Sequence[int]) -> torch.Tensor:
    """Return raw internal message sum [B,T,k,4] using exact historical scheduling."""
    ordered = tuple(sorted(int(n) for n in nodes)); node_set = set(ordered)
    B, T = trace.outputs.shape[:2]
    out = torch.zeros(B, T, len(ordered), trace.outputs.shape[-1], dtype=trace.outputs.dtype)
    node_to_local = {n:i for i,n in enumerate(ordered)}
    prev = previous_outputs(trace.outputs)
    for edge_index, edge in enumerate(model.graph.edges):
        if int(edge.src) not in node_set or int(edge.dst) not in node_set:
            continue
        dst = node_to_local[int(edge.dst)]
        source = prev[:, :, int(edge.src)] if bool(edge.recurrent) else trace.outputs[:, :, int(edge.src)]
        out[:, :, dst] += model.base.edge_weights[edge_index].detach().cpu() * source
    mask = active_mask(trace).to(out.dtype).unsqueeze(-1).unsqueeze(-1)
    return out * mask


def external_raw_messages(model, trace: FullProbeTrace, nodes: Sequence[int]) -> torch.Tensor:
    ordered = tuple(sorted(int(n) for n in nodes))
    total = trace.messages[:, :, list(ordered), :]
    return total - internal_raw_messages(model, trace, ordered)


def boundary_streams(model, trace: FullProbeTrace, nodes: Sequence[int]) -> dict[str, torch.Tensor]:
    ordered = tuple(sorted(int(n) for n in nodes))
    return {
        "observations": trace.observations,
        "lengths": trace.lengths,
        "external_messages": external_raw_messages(model, trace, ordered),
        "global_terms": trace.global_terms[:, :, list(ordered), :] + trace.matched_terms[:, :, list(ordered), :],
        "gates": trace.gates,
    }
