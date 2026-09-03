from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .graph import GraphSpec
from .resource_accounting import ResourceAccounting, WallTimer
from .seeds import deterministic_int
from .substrate import NeutralGraphModel, clone_with_state
from .task_interface import Episode, OBS_DIM, TaskFamily


@dataclass
class Evaluation:
    loss: float
    success_rate: float
    successes: list[bool]
    predictions: list[float]
    targets: list[float]

    def to_dict(self) -> dict:
        return {
            "loss": self.loss,
            "success_rate": self.success_rate,
            "successes": self.successes,
            "predictions": self.predictions,
            "targets": self.targets,
        }


@dataclass
class TrainedCandidate:
    graph: GraphSpec
    model: NeutralGraphModel
    development: Evaluation
    validation: Evaluation
    resources: ResourceAccounting


def episodes_to_batch(episodes: list[Episode]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_steps = max(len(episode.observations) for episode in episodes)
    observations = np.zeros((len(episodes), max_steps, OBS_DIM), dtype=np.float32)
    lengths = np.zeros(len(episodes), dtype=np.int64)
    targets = np.zeros(len(episodes), dtype=np.float32)
    for index, episode in enumerate(episodes):
        length = len(episode.observations)
        observations[index, :length] = episode.observations
        lengths[index] = length
        targets[index] = episode.target
    return torch.from_numpy(observations), torch.from_numpy(lengths), torch.from_numpy(targets)


def evaluate_model(model: NeutralGraphModel, task: TaskFamily, episodes: list[Episode]) -> Evaluation:
    model.eval()
    observations, lengths, targets = episodes_to_batch(episodes)
    with torch.no_grad():
        predictions = model(observations, lengths)
        loss = torch.mean((predictions - targets) ** 2).item()
    pred_values = [float(value) for value in predictions.tolist()]
    target_values = [float(value) for value in targets.tolist()]
    successes = [bool(task.success(pred, target)) for pred, target in zip(pred_values, target_values)]
    return Evaluation(
        loss=float(loss),
        success_rate=float(np.mean(successes)),
        successes=successes,
        predictions=pred_values,
        targets=target_values,
    )


def train_graph(
    graph: GraphSpec,
    task: TaskFamily,
    train_seeds: list[int],
    validation_seeds: list[int],
    *,
    run_seed: int,
    state_dim: int = 4,
    message_dim: int = 4,
    steps: int = 24,
    learning_rate: float = 0.02,
    weight_decay: float = 1e-4,
    training_scope: str = "readout_only_adamw",
) -> TrainedCandidate:
    graph.validate()
    init_seed = deterministic_int("train", graph.graph_id, task.name, run_seed)
    torch.manual_seed(init_seed)
    np.random.seed(init_seed % (2**32 - 1))
    model = NeutralGraphModel(graph, obs_dim=OBS_DIM, state_dim=state_dim, message_dim=message_dim)
    if training_scope == "readout_only_adamw":
        optimized_parameters = list(model.readout.parameters())
    elif training_scope == "full_adamw":
        optimized_parameters = list(model.parameters())
    else:
        raise ValueError(f"unknown training scope: {training_scope}")
    optimizer = torch.optim.AdamW(optimized_parameters, lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_episodes = [task.generate(seed, "development") for seed in train_seeds]
    validation_episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(train_episodes)
    unique_env_steps = sum(len(episode.observations) for episode in train_episodes + validation_episodes)
    resources = ResourceAccounting(
        candidate_evaluations=1,
        optimizer_steps=0,
        environment_steps=unique_env_steps,
        examples_processed=0,
        model_fits=1,
        peak_cells=len(graph.cells),
        peak_edges=len(graph.edges),
        final_cells=len(graph.cells),
        final_edges=len(graph.edges),
        parameter_count=model.parameter_count(),
    )
    with WallTimer() as timer:
        model.train()
        if training_scope == "readout_only_adamw":
            with torch.no_grad():
                _, trace = model(observations, lengths, return_trace=True)
                train_features = trace.states[:, -1].reshape(len(train_episodes), -1).detach()
            for _ in range(int(steps)):
                optimizer.zero_grad(set_to_none=True)
                predictions = torch.tanh(model.readout(train_features)).squeeze(-1)
                loss = loss_fn(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.readout.parameters(), 5.0)
                optimizer.step()
                resources.optimizer_steps += 1
                resources.examples_processed += len(train_episodes)
        else:
            for _ in range(int(steps)):
                optimizer.zero_grad(set_to_none=True)
                predictions = model(observations, lengths)
                loss = loss_fn(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                resources.optimizer_steps += 1
                resources.examples_processed += len(train_episodes)
        development = evaluate_model(model, task, train_episodes)
        validation = evaluate_model(model, task, validation_episodes)
    resources.wall_seconds = timer.seconds
    return TrainedCandidate(graph=graph.clone(), model=model, development=development, validation=validation, resources=resources)


def refine_candidate_full_adamw(
    candidate: TrainedCandidate,
    task: TaskFamily,
    train_seeds: list[int],
    validation_seeds: list[int],
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
) -> TrainedCandidate:
    """Refine all continuous parameters of an already-selected graph.

    The topology is frozen. This isolates whether the initial competence
    failure came from readout-only parameter adaptation rather than from the
    structural representation itself.
    """
    model = clone_with_state(candidate.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_episodes = [task.generate(seed, "development") for seed in train_seeds]
    validation_episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(train_episodes)
    unique_env_steps = sum(len(episode.observations) for episode in train_episodes + validation_episodes)
    resources = ResourceAccounting(
        candidate_evaluations=1,
        optimizer_steps=0,
        environment_steps=unique_env_steps,
        examples_processed=0,
        model_fits=1,
        peak_cells=len(candidate.graph.cells),
        peak_edges=len(candidate.graph.edges),
        final_cells=len(candidate.graph.cells),
        final_edges=len(candidate.graph.edges),
        parameter_count=model.parameter_count(),
    )
    with WallTimer() as timer:
        model.train()
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            predictions = model(observations, lengths)
            loss = loss_fn(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            resources.optimizer_steps += 1
            resources.examples_processed += len(train_episodes)
        development = evaluate_model(model, task, train_episodes)
        validation = evaluate_model(model, task, validation_episodes)
    resources.wall_seconds = timer.seconds
    return TrainedCandidate(
        graph=candidate.graph.clone(),
        model=model,
        development=development,
        validation=validation,
        resources=resources,
    )
