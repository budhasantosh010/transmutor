from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .graph import CellSpec, EdgeSpec, GraphSpec
from .metrics import binary_summary
from .seeds import rng
from .substrate import NeutralGraphModel, clone_with_state
from .task_interface import TaskFamily
from .trainer import episodes_to_batch, evaluate_model


def oracle_validation(task: TaskFamily, seeds: list[int], split: str = "validation") -> dict:
    outcomes = []
    errors = []
    for seed in seeds:
        episode = task.generate(seed, split)
        prediction = float(task.oracle(episode))
        outcomes.append(task.success(prediction, episode.target))
        errors.append(abs(prediction - episode.target))
    return {"binary": binary_summary(outcomes), "max_absolute_error": float(max(errors) if errors else 0.0)}


def first_observation_leakage(tasks: list[TaskFamily], seeds: list[int], *, test_fraction: float = 0.35, seed: int = 837) -> dict:
    features: list[np.ndarray] = []
    labels: list[str] = []
    for task in tasks:
        for episode_seed in seeds:
            episode = task.generate(episode_seed, "development")
            features.append(episode.observations[0])
            labels.append(task.name)
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_fraction, random_state=seed, stratify=y)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    accuracy = float(accuracy_score(y_test, prediction))
    return {"accuracy": accuracy, "n_train": int(len(y_train)), "n_test": int(len(y_test)), "chance": 1.0 / len(tasks)}


def random_matched_graph(reference: GraphSpec, seed: int) -> GraphSpec:
    random = rng("matched_graph", reference.graph_id, seed)
    n = len(reference.cells)
    e = len(reference.edges)
    recurrent_count = sum(edge.recurrent for edge in reference.edges)
    cells = [CellSpec(i, param_seed=random.randrange(1, 2**20), birth_generation=0) for i in range(n)]
    all_pairs = [(a, b) for a in range(n) for b in range(n)]
    random.shuffle(all_pairs)
    chosen = all_pairs[:e]
    recurrent_flags = [True] * recurrent_count + [False] * max(0, e - recurrent_count)
    random.shuffle(recurrent_flags)
    edges = [
        EdgeSpec(src, dst, weight=random.uniform(-1.0, 1.0), recurrent=recurrent_flags[index])
        for index, (src, dst) in enumerate(chosen)
    ]
    graph = GraphSpec(
        cells=cells,
        edges=edges,
        input_access=reference.input_access,
        generation=0,
        parent_id="RANDOM_MATCHED",
    )
    graph.validate()
    return graph


def generic_dynamic_descriptors(model: NeutralGraphModel, episodes) -> dict[str, float]:
    observations, lengths, _ = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        _, trace = model(observations, lengths, return_trace=True)
    states = trace.states.detach().cpu().numpy()  # B,T,N,D
    active_values = states.reshape(-1, states.shape[2], states.shape[3])
    mean_activation = float(np.mean(np.abs(active_values)))
    activation_variance = float(np.var(active_values))
    final_abs = float(np.mean(np.abs(states[:, -1])))
    if states.shape[1] >= 2:
        left = states[:, :-1].reshape(-1)
        right = states[:, 1:].reshape(-1)
        if np.std(left) > 1e-9 and np.std(right) > 1e-9:
            autocorr = float(np.corrcoef(left, right)[0, 1])
        else:
            autocorr = 0.0
    else:
        autocorr = 0.0
    per_cell_std = np.std(states, axis=(0, 1, 3))
    effective_path_diversity = float(np.mean(per_cell_std > 0.05))
    obs_first = observations[:, :, 0].numpy()
    cell_energy = np.mean(np.abs(states), axis=(2, 3))
    flat_obs = obs_first.reshape(-1)
    flat_energy = cell_energy.reshape(-1)
    if np.std(flat_obs) > 1e-9 and np.std(flat_energy) > 1e-9:
        input_sensitivity = float(abs(np.corrcoef(flat_obs, flat_energy)[0, 1]))
    else:
        input_sensitivity = 0.0
    return {
        "mean_absolute_activation": mean_activation,
        "activation_variance": activation_variance,
        "final_state_persistence": final_abs,
        "state_autocorrelation": autocorr,
        "effective_path_diversity": effective_path_diversity,
        "generic_input_sensitivity": input_sensitivity,
    }


def graph_and_dynamic_descriptors(model: NeutralGraphModel, task: TaskFamily, seeds: list[int], split: str = "validation") -> dict[str, float]:
    episodes = [task.generate(seed, split) for seed in seeds]
    result = model.graph.descriptors()
    result.update(generic_dynamic_descriptors(model, episodes))
    return result


def differentiation_classifier(records: list[dict], *, seed: int = 837, permutations: int = 499) -> dict:
    if not records:
        raise ValueError("no differentiation records")
    feature_names = sorted(records[0]["descriptors"])
    x = np.asarray([[row["descriptors"][name] for name in feature_names] for row in records], dtype=np.float64)
    y = np.asarray([row["family"] for row in records])
    run_ids = np.asarray([int(row["run_index"]) for row in records])
    test_mask = run_ids >= 20
    train_mask = ~test_mask
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    classifier.fit(x[train_mask], y[train_mask])
    observed = float(accuracy_score(y[test_mask], classifier.predict(x[test_mask])))
    generator = np.random.default_rng(seed)
    extreme = 0
    for _ in range(permutations):
        permuted = y.copy()
        generator.shuffle(permuted)
        candidate = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
        candidate.fit(x[train_mask], permuted[train_mask])
        score = float(accuracy_score(permuted[test_mask], candidate.predict(x[test_mask])))
        if score >= observed - 1e-12:
            extreme += 1
    return {
        "accuracy": observed,
        "p_value": (extreme + 1) / (permutations + 1),
        "permutations": permutations,
        "feature_names": feature_names,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def disable_cells(model: NeutralGraphModel, task: TaskFamily, episodes, cells: set[int]):
    model.eval()
    observations, lengths, targets = episodes_to_batch(episodes)
    with torch.no_grad():
        predictions = model(observations, lengths, disabled_cells=set(cells))
    losses = ((predictions - targets) ** 2).detach().cpu().numpy()
    success = np.asarray([task.success(float(p), float(t)) for p, t in zip(predictions.tolist(), targets.tolist())], dtype=bool)
    return losses, success


def randomized_cell_parameters(model: NeutralGraphModel, cells: set[int], seed: int) -> NeutralGraphModel:
    clone = clone_with_state(model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for index in cells:
            for parameter_list in [clone.cell_ws, clone.cell_wm, clone.cell_wx, clone.cell_wo]:
                tensor = parameter_list[index]
                tensor.copy_(torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype) * 0.25)
            clone.cell_b[index].zero_()
    return clone


def _masked_trace_values(tensor: torch.Tensor, lengths: torch.Tensor) -> np.ndarray:
    """Flatten valid [B,T,N,D] trace positions without padded timesteps."""
    values = tensor.detach().cpu().numpy()
    valid_chunks = []
    for batch_index, length in enumerate(lengths.detach().cpu().tolist()):
        valid_chunks.append(values[batch_index, : int(length)])
    if not valid_chunks:
        return np.zeros((0,) + values.shape[2:], dtype=values.dtype)
    return np.concatenate(valid_chunks, axis=0)


def representation_diagnostics(
    model: NeutralGraphModel,
    task: TaskFamily,
    episodes,
    *,
    forward_options: dict | None = None,
    include_cell_ablations: bool = True,
) -> dict:
    """Generic representation diagnostics with no task-semantic scores."""
    options = dict(forward_options or {})
    observations, lengths, _ = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        _, trace = model(observations, lengths, return_trace=True, **options)
    states = _masked_trace_values(trace.states, lengths)  # [valid_t,N,D]
    messages = _masked_trace_values(trace.messages, lengths)
    recurrent_terms = _masked_trace_values(trace.recurrent_terms, lengths)
    message_terms = _masked_trace_values(trace.message_terms, lengths)
    input_terms = _masked_trace_values(trace.input_terms, lengths)
    n = len(model.graph.cells)

    flattened_cells = [states[:, cell, :].reshape(-1) for cell in range(n)] if states.size else []
    correlations = []
    for left in range(n):
        for right in range(left + 1, n):
            a = flattened_cells[left]
            b = flattened_cells[right]
            if a.size and np.std(a) > 1e-9 and np.std(b) > 1e-9:
                correlations.append(abs(float(np.corrcoef(a, b)[0, 1])))
            else:
                correlations.append(0.0)

    autocorrelations = []
    for cell in range(n):
        left_values = []
        right_values = []
        raw = trace.states.detach().cpu().numpy()
        for batch_index, length in enumerate(lengths.detach().cpu().tolist()):
            length = int(length)
            if length >= 2:
                left_values.append(raw[batch_index, : length - 1, cell, :].reshape(-1))
                right_values.append(raw[batch_index, 1:length, cell, :].reshape(-1))
        if left_values:
            a = np.concatenate(left_values)
            b = np.concatenate(right_values)
            if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                autocorrelations.append(float(np.corrcoef(a, b)[0, 1]))
            else:
                autocorrelations.append(0.0)
        else:
            autocorrelations.append(0.0)

    per_cell_variance = np.var(states, axis=(0, 2)) if states.size else np.zeros(n)
    correlation_array = np.asarray(correlations, dtype=float)
    result = {
        "input_edge_count": model.input_edge_count,
        "effective_input_density": float(model.input_edge_count / max(1, model.obs_dim * n)),
        "internal_message_edge_count": model.internal_message_edge_count,
        "mean_cell_activation": float(np.mean(np.abs(states))) if states.size else 0.0,
        "cell_activation_variance": float(np.var(states)) if states.size else 0.0,
        "state_saturation_fraction": float(np.mean(np.abs(states) >= 0.95)) if states.size else 0.0,
        "mean_pairwise_state_corr": float(np.mean(correlation_array)) if correlation_array.size else 0.0,
        "median_pairwise_state_corr": float(np.median(correlation_array)) if correlation_array.size else 0.0,
        "p90_pairwise_state_corr": float(np.quantile(correlation_array, 0.90)) if correlation_array.size else 0.0,
        "cell_state_autocorrelation": float(np.mean(autocorrelations)) if autocorrelations else 0.0,
        "message_magnitude": float(np.mean(np.abs(messages))) if messages.size else 0.0,
        "raw_input_contribution_magnitude": float(np.mean(np.abs(input_terms))) if input_terms.size else 0.0,
        "internal_message_contribution_magnitude": float(np.mean(np.abs(message_terms))) if message_terms.size else 0.0,
        "recurrent_state_contribution_magnitude": float(np.mean(np.abs(recurrent_terms))) if recurrent_terms.size else 0.0,
        "effective_active_cell_count": int(np.sum(per_cell_variance > 1e-4)),
        "diagnostic_episode_count": len(episodes),
    }

    if include_cell_ablations:
        baseline = evaluate_model(model, task, episodes, forward_options=options)
        raw_effects = []
        message_effects = []
        for cell in range(n):
            raw_options = dict(options)
            raw_disabled = set(raw_options.get("disabled_raw_input_cells", set()))
            raw_disabled.add(cell)
            raw_options["disabled_raw_input_cells"] = raw_disabled
            raw_eval = evaluate_model(model, task, episodes, forward_options=raw_options)
            raw_effects.append(float(raw_eval.loss - baseline.loss))

            message_options = dict(options)
            message_disabled = set(message_options.get("disabled_message_cells", set()))
            message_disabled.add(cell)
            message_options["disabled_message_cells"] = message_disabled
            message_eval = evaluate_model(model, task, episodes, forward_options=message_options)
            message_effects.append(float(message_eval.loss - baseline.loss))

        mean_raw = float(np.mean(raw_effects)) if raw_effects else 0.0
        mean_message = float(np.mean(message_effects)) if message_effects else 0.0
        denominator = mean_raw + mean_message + 1e-12
        result.update(
            {
                "raw_ablation_effects": raw_effects,
                "message_ablation_effects": message_effects,
                "mean_raw_ablation_effect": mean_raw,
                "median_raw_ablation_effect": float(np.median(raw_effects)) if raw_effects else 0.0,
                "mean_message_ablation_effect": mean_message,
                "median_message_ablation_effect": float(np.median(message_effects)) if message_effects else 0.0,
                "message_dependency_ratio": float(mean_message / denominator) if abs(denominator) > 1e-12 else 0.0,
            }
        )
    return result
