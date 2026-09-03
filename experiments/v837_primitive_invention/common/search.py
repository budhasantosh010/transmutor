from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .evaluator import random_matched_graph
from .graph import GraphSpec, initial_graph
from .mutations import mutate
from .resource_accounting import ResourceAccounting, WallTimer
from .seeds import cyclic_seeds, deterministic_int, frozen_gates
from .task_interface import TaskFamily
from .trainer import TrainedCandidate, train_graph


@dataclass
class SearchResult:
    task_family: str
    run_index: int
    best: TrainedCandidate
    random_control: TrainedCandidate
    solved_development: bool
    solved_validation: bool
    generations_used: int
    candidate_summaries: list[dict[str, Any]]
    mutation_counts: dict[str, int]
    resources: ResourceAccounting
    train_seeds: list[int]
    validation_seeds: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_family": self.task_family,
            "run_index": self.run_index,
            "best_graph": self.best.graph.to_dict(),
            "development": self.best.development.to_dict(),
            "validation": self.best.validation.to_dict(),
            "random_matched": {
                "graph": self.random_control.graph.to_dict(),
                "development": self.random_control.development.to_dict(),
                "validation": self.random_control.validation.to_dict(),
                "resources": self.random_control.resources.to_dict(),
            },
            "solved_development": self.solved_development,
            "solved_validation": self.solved_validation,
            "generations_used": self.generations_used,
            "candidate_summaries": self.candidate_summaries,
            "mutation_counts": self.mutation_counts,
            "resources": self.resources.to_dict(),
            "train_seeds": self.train_seeds,
            "validation_seeds": self.validation_seeds,
        }


def candidate_fitness(candidate: TrainedCandidate, lambda_cells: float, lambda_edges: float) -> float:
    return (
        float(candidate.validation.loss)
        + lambda_cells * len(candidate.graph.cells)
        + lambda_edges * len(candidate.graph.edges)
    )


def structural_search(task: TaskFamily, run_index: int, *, overrides: dict[str, Any] | None = None) -> SearchResult:
    gates = frozen_gates()
    search = dict(gates["search"])
    if overrides:
        search.update(overrides)
    population_size = int(search["population"])
    max_generations = int(search["max_generations"])
    offspring_per_generation = int(search["offspring_per_generation"])
    lambda_cells = float(search["lambda_cells"])
    lambda_edges = float(search["lambda_edges"])
    train_steps = int(search["candidate_train_steps"])
    training_scope = str(search.get("parameter_training_scope", "readout_only_adamw"))
    learning_rate = float(search["learning_rate"])
    weight_decay = float(search["weight_decay"])
    development_count = int(search["development_episodes_per_candidate"])
    validation_count = int(search["validation_episodes_per_candidate"])
    state_dim = int(gates["substrate"]["state_dim"])
    message_dim = int(gates["substrate"]["message_dim"])
    max_cells = int(gates["substrate"]["max_cells"])
    max_edges = int(gates["substrate"]["max_edges"])
    train_seeds = cyclic_seeds("development", development_count, offset=run_index * 31)
    validation_seeds = cyclic_seeds("validation", validation_count, offset=run_index * 17)
    run_seed = deterministic_int("v837-search", task.name, run_index, overrides or {})
    cache: dict[str, TrainedCandidate] = {}
    candidate_summaries: list[dict[str, Any]] = []
    mutation_counts: dict[str, int] = {}
    total = ResourceAccounting(
        max_candidate_budget=population_size + offspring_per_generation * max_generations,
        max_optimizer_step_budget=(population_size + offspring_per_generation * max_generations) * train_steps,
    )

    def evaluate(graph: GraphSpec) -> TrainedCandidate:
        if graph.graph_id in cache:
            return cache[graph.graph_id]
        candidate = train_graph(
            graph,
            task,
            train_seeds,
            validation_seeds,
            run_seed=run_seed,
            state_dim=state_dim,
            message_dim=message_dim,
            steps=train_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            training_scope=training_scope,
        )
        cache[graph.graph_id] = candidate
        total.merge(candidate.resources)
        candidate_summaries.append(
            {
                "graph_id": graph.graph_id,
                "generation": graph.generation,
                "cells": len(graph.cells),
                "edges": len(graph.edges),
                "development_loss": candidate.development.loss,
                "development_success": candidate.development.success_rate,
                "validation_loss": candidate.validation.loss,
                "validation_success": candidate.validation.success_rate,
                "fitness": candidate_fitness(candidate, lambda_cells, lambda_edges),
            }
        )
        return candidate

    base = initial_graph()
    population_graphs = [base]
    parent = base
    for index in range(population_size - 1):
        child, op = mutate(parent, deterministic_int(run_seed, "initial", index), max_cells=max_cells, max_edges=max_edges)
        mutation_counts[op] = mutation_counts.get(op, 0) + 1
        population_graphs.append(child)
        parent = child if child.graph_id != parent.graph_id else base

    with WallTimer() as timer:
        population = [evaluate(graph) for graph in population_graphs]
        population.sort(key=lambda item: candidate_fitness(item, lambda_cells, lambda_edges))
        generations_used = 0
        for generation in range(max_generations + 1):
            population.sort(key=lambda item: candidate_fitness(item, lambda_cells, lambda_edges))
            best = population[0]
            if best.development.success_rate >= gates["v837"]["development_success_rate_per_family"] and best.validation.success_rate >= gates["v837"]["heldout_validation_success_rate_per_family"]:
                generations_used = generation
                break
            if generation >= max_generations:
                generations_used = max_generations
                break
            parent_pool = population[: max(2, population_size // 2)]
            offspring: list[TrainedCandidate] = []
            for child_index in range(offspring_per_generation):
                parent_candidate = parent_pool[(generation * offspring_per_generation + child_index) % len(parent_pool)]
                child_graph, op = mutate(
                    parent_candidate.graph,
                    deterministic_int(run_seed, "offspring", generation, child_index, parent_candidate.graph.graph_id),
                    max_cells=max_cells,
                    max_edges=max_edges,
                )
                mutation_counts[op] = mutation_counts.get(op, 0) + 1
                total.mutation_count += 1
                total.search_expansions += 1
                offspring.append(evaluate(child_graph))
            population = sorted(population + offspring, key=lambda item: candidate_fitness(item, lambda_cells, lambda_edges))[:population_size]
        best = sorted(population, key=lambda item: candidate_fitness(item, lambda_cells, lambda_edges))[0]
    total.wall_seconds = timer.seconds
    total.final_cells = len(best.graph.cells)
    total.final_edges = len(best.graph.edges)
    random_graph = random_matched_graph(best.graph, deterministic_int(run_seed, "random-control"))
    random_control = train_graph(
        random_graph,
        task,
        train_seeds,
        validation_seeds,
        run_seed=deterministic_int(run_seed, "random-control-train"),
        state_dim=state_dim,
        message_dim=message_dim,
        steps=train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        training_scope=training_scope,
    )
    return SearchResult(
        task_family=task.name,
        run_index=run_index,
        best=best,
        random_control=random_control,
        solved_development=best.development.success_rate >= gates["v837"]["development_success_rate_per_family"],
        solved_validation=best.validation.success_rate >= gates["v837"]["heldout_validation_success_rate_per_family"],
        generations_used=generations_used,
        candidate_summaries=candidate_summaries,
        mutation_counts=mutation_counts,
        resources=total,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
    )
