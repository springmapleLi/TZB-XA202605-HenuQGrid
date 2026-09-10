"""Budget-matched classical placement-refinement baselines."""
from __future__ import annotations

from math import exp
from random import Random
from typing import Sequence

from Circuit_Hardware_Model import AssignmentMetrics
from Hamming_Trust_Region import structured_candidates
from Multi_Fidelity_Evaluator import NativeQiskitPlacementEvaluator


def no_correction(assignment: Sequence[int], evaluator: NativeQiskitPlacementEvaluator) -> AssignmentMetrics:
    return evaluator.evaluate(assignment)


def greedy_correction_search(
    assignment: Sequence[int],
    evaluator: NativeQiskitPlacementEvaluator,
    budget: int = 128,
) -> AssignmentMetrics:
    best = evaluator.evaluate(assignment)
    used = 1
    while used < budget:
        active = range(len(best.assignment))
        candidates = []
        for candidate in structured_candidates(best.assignment, evaluator.hardware, active):
            candidates.append(evaluator.evaluate(candidate))
            used += 1
            if used >= budget:
                break
        if not candidates:
            break
        winner = min(candidates, key=lambda value: (value.objective, value.assignment))
        if winner.objective >= best.objective:
            break
        best = winner
    return best


def random_action_search(
    assignment: Sequence[int],
    evaluator: NativeQiskitPlacementEvaluator,
    budget: int = 128,
    seed: int = 0,
) -> AssignmentMetrics:
    rng = Random(seed)
    current = evaluator.evaluate(assignment)
    best = current
    for _ in range(max(0, budget - 1)):
        candidates = list(structured_candidates(current.assignment, evaluator.hardware, range(len(current.assignment))))
        if not candidates:
            break
        current = evaluator.evaluate(rng.choice(candidates))
        if current.objective < best.objective:
            best = current
    return best


def simulated_annealing_search(
    assignment: Sequence[int],
    evaluator: NativeQiskitPlacementEvaluator,
    budget: int = 128,
    seed: int = 0,
    initial_temperature: float = 5.0,
) -> AssignmentMetrics:
    rng = Random(seed)
    current = evaluator.evaluate(assignment)
    best = current
    for step in range(1, max(1, budget)):
        candidates = list(structured_candidates(current.assignment, evaluator.hardware, range(len(current.assignment))))
        if not candidates:
            break
        trial = evaluator.evaluate(rng.choice(candidates))
        temperature = max(1e-9, initial_temperature * (1.0 - step / max(1, budget)))
        delta = trial.objective - current.objective
        if delta <= 0 or rng.random() < exp(-delta / temperature):
            current = trial
        if current.objective < best.objective:
            best = current
    return best


def fm_style_refinement(
    assignment: Sequence[int],
    evaluator: NativeQiskitPlacementEvaluator,
    budget: int = 128,
) -> AssignmentMetrics:
    """Compact FM-style best-swap pass under a full-evaluation budget."""
    return greedy_correction_search(assignment, evaluator, budget)
