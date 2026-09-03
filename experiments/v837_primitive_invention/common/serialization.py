from __future__ import annotations

import json
from pathlib import Path

import torch

from .graph import GraphSpec
from .substrate import NeutralGraphModel


def write_json(path: str | Path, data: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_model_bundle(path: str | Path, model: NeutralGraphModel, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "graph": model.graph.to_dict(),
            "obs_dim": model.obs_dim,
            "state_dim": model.state_dim,
            "message_dim": model.message_dim,
            "state_update_mode": model.state_update_mode,
            "alpha_init": model.alpha_init,
            "interaction_mode": model.interaction_mode,
            "interaction_rank": model.interaction_rank,
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_model_bundle(path: str | Path) -> tuple[NeutralGraphModel, dict]:
    bundle = torch.load(Path(path), map_location="cpu", weights_only=False)
    graph = GraphSpec.from_dict(bundle["graph"])
    model = NeutralGraphModel(
        graph,
        obs_dim=bundle["obs_dim"],
        state_dim=bundle["state_dim"],
        message_dim=bundle["message_dim"],
        state_update_mode=bundle.get("state_update_mode", "direct"),
        alpha_init=float(bundle.get("alpha_init", 0.5)),
        interaction_mode=bundle.get("interaction_mode", "none"),
        interaction_rank=int(bundle.get("interaction_rank", 2)),
    )
    model.load_state_dict(bundle["state_dict"])
    return model, dict(bundle.get("metadata", {}))
