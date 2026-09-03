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
    graph = GraphSpec(cells=cells, edges=edges, generation=0, parent_id="RANDOM_MATCHED")
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
