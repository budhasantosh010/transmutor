from .conditional_routing import ConditionalRoutingTask
from .delayed_recall import DelayedRecallTask
from .iterative_state import IterativeStateTask
from .partial_observation import PartialObservationTask
from .variable_composition import VariableCompositionTask


def all_tasks():
    return [
        DelayedRecallTask(),
        ConditionalRoutingTask(),
        IterativeStateTask(),
        VariableCompositionTask(),
        PartialObservationTask(),
    ]


def task_by_name(name: str):
    for task in all_tasks():
        if task.name == name:
            return task
    raise KeyError(name)
