from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from .resource_accounting import ResourceAccounting, WallTimer
from .task_interface import Episode, TaskFamily
from .trainer import Evaluation, episodes_to_batch


@dataclass
class ReferenceTrainingResult:
    model: nn.Module
    development: Evaluation
    validation: Evaluation
    resources: ResourceAccounting
    learning_curve: list[dict]


def evaluate_sequence_model(model: nn.Module, task: TaskFamily, episodes: list[Episode]) -> Evaluation:
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


def _state_diagnostics(model: nn.Module, episodes: list[Episode]) -> dict:
    observations, lengths, _ = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        _, trace = model(observations, lengths, return_trace=True)
    states = trace.states
    if states.ndim == 4:
        states = states.reshape(states.shape[0], states.shape[1], -1)
    if states.ndim != 3:
        raise ValueError("state trace must reduce to [B,T,D]")
    active = torch.arange(states.shape[1]).view(1, -1) < lengths.view(-1, 1)
    active_states = states[active]
    if not active_states.numel():
        return {"state_norm": 0.0, "activation_saturation": 0.0}
    output = {
        "state_norm": float(torch.linalg.vector_norm(active_states, dim=-1).mean().item()),
        "activation_saturation": float((torch.abs(active_states) >= 0.95).float().mean().item()),
    }
    candidates = getattr(trace, "candidates", None)
    if candidates is not None:
        active_candidates = candidates[active]
        output["candidate_state_norm"] = float(torch.linalg.vector_norm(active_candidates, dim=-1).mean().item())
        output["candidate_saturation"] = float((torch.abs(active_candidates) >= 0.95).float().mean().item())
    updates = getattr(trace, "updates", None)
    resets = getattr(trace, "resets", None)
    if updates is not None:
        active_updates = updates[active]
        output["update_coefficient_mean"] = float(active_updates.mean().item())
        output["update_coefficient_variance"] = float(active_updates.var(unbiased=False).item())
    if resets is not None:
        active_resets = resets[active]
        output["candidate_condition_mean"] = float(active_resets.mean().item())
        output["candidate_condition_variance"] = float(active_resets.var(unbiased=False).item())
    return output


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(torch.sum(parameter.grad.detach() ** 2))
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).item())


def matched_budget_signature(*, optimizer: str, optimizer_steps: int, train_episodes: int, validation_episodes: int, learning_rate: float, weight_decay: float, gradient_clip: float) -> dict:
    return {
        "optimizer": optimizer,
        "optimizer_steps": int(optimizer_steps),
        "train_episodes": int(train_episodes),
        "validation_episodes": int(validation_episodes),
        "examples_processed": int(optimizer_steps) * int(train_episodes),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "gradient_clip": float(gradient_clip),
        "batch_construction": "single full development episode batch reused for each optimizer step",
    }


def train_sequence_model(*, model_factory: Callable[[], nn.Module], task: TaskFamily, train_seeds: list[int], validation_seeds: list[int], initialization_seed: int, steps: int, learning_rate: float, weight_decay: float, gradient_clip: float = 5.0, curve_steps: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192)) -> ReferenceTrainingResult:
    """Train a neutral/reference sequence model under one matched full-batch regime."""
    torch.manual_seed(int(initialization_seed))
    np.random.seed(int(initialization_seed) % (2**32 - 1))
    model = model_factory()
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()
    train_episodes = [task.generate(seed, "development") for seed in train_seeds]
    validation_episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(train_episodes)
    unique_env_steps = sum(len(episode.observations) for episode in train_episodes + validation_episodes)
    parameter_count = int(sum(parameter.numel() for parameter in parameters))
    parameter_bytes = int(sum(parameter.numel() * parameter.element_size() for parameter in parameters))
    resources = ResourceAccounting(
        candidate_evaluations=1,
        optimizer_steps=0,
        environment_steps=unique_env_steps,
        examples_processed=0,
        model_fits=1,
        input_edges=int(getattr(model, "input_edge_count", 0)),
        internal_message_edges=int(getattr(model, "internal_message_edge_count", 0)),
        parameter_count=parameter_count,
        model_parameter_bytes=parameter_bytes,
    )
    requested_curve_steps = sorted(set(int(step) for step in curve_steps if 0 <= int(step) <= int(steps)))
    if int(steps) not in requested_curve_steps:
        requested_curve_steps.append(int(steps))
    curve: list[dict] = []
    latest_gradient_norm = 0.0

    def record(step: int) -> None:
        development = evaluate_sequence_model(model, task, train_episodes)
        validation = evaluate_sequence_model(model, task, validation_episodes)
        state_diag = _state_diagnostics(model, validation_episodes)
        resources.forward_calls += 3
        curve.append({
            "step": int(step),
            "training_loss": development.loss,
            "training_success": development.success_rate,
            "validation_loss": validation.loss,
            "validation_success": validation.success_rate,
            "gradient_norm": float(latest_gradient_norm),
            **state_diag,
        })

    with WallTimer() as timer:
        if 0 in requested_curve_steps:
            record(0)
        for step in range(1, int(steps) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predictions = model(observations, lengths)
            resources.forward_calls += 1
            loss = loss_fn(predictions, targets)
            loss.backward()
            latest_gradient_norm = _gradient_norm(parameters)
            torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip))
            optimizer.step()
            resources.optimizer_steps += 1
            resources.examples_processed += len(train_episodes)
            if step in requested_curve_steps:
                record(step)
        development = evaluate_sequence_model(model, task, train_episodes)
        validation = evaluate_sequence_model(model, task, validation_episodes)
        resources.forward_calls += 2
    resources.wall_seconds = timer.seconds
    resources.cpu_seconds = timer.cpu_seconds
    return ReferenceTrainingResult(
        model=model,
        development=development,
        validation=validation,
        resources=resources,
        learning_curve=curve,
    )
