"""Core circuit and heterogeneous multi-QPU hardware data structures.

This file is a compact extraction of ``residual_dqc/model.py`` and
``residual_dqc/circuits.py`` from the research repository.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from random import Random
from typing import Iterable, Sequence


@dataclass(frozen=True)
class LayeredCircuit:
    """A two-qubit interaction circuit grouped into executable layers."""

    num_qubits: int
    layers: tuple[tuple[tuple[int, int], ...], ...]

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError("num_qubits must be positive")
        for layer in self.layers:
            used: set[int] = set()
            for left, right in layer:
                if left == right or not 0 <= left < self.num_qubits or not 0 <= right < self.num_qubits:
                    raise ValueError(f"invalid interaction {(left, right)}")
                if left in used or right in used:
                    raise ValueError("a qubit may occur only once in a circuit layer")
                used.update((left, right))

    @property
    def interactions(self) -> tuple[tuple[int, int], ...]:
        return tuple(edge for layer in self.layers for edge in layer)

    @property
    def interaction_depth(self) -> int:
        return len(self.layers)

    @property
    def two_qubit_gate_count(self) -> int:
        return sum(len(layer) for layer in self.layers)

    def interaction_weights(self) -> dict[tuple[int, int], float]:
        weights: dict[tuple[int, int], float] = {}
        for left, right in self.interactions:
            edge = tuple(sorted((left, right)))
            weights[edge] = weights.get(edge, 0.0) + 1.0
        return weights


@dataclass(frozen=True)
class CostProfile:
    """Weights of J = w_R R + w_E E + w_L L."""

    remote_rounds: float = 10.0
    epr_hops: float = 1.0
    local_added_2q: float = 1.0


@dataclass(frozen=True)
class HardwareModel:
    """Labelled QPUs, capacities, local couplings and network resources."""

    capacities: tuple[int, ...]
    network_edges: tuple[tuple[int, int], ...]
    local_edges: tuple[tuple[tuple[int, int], ...], ...]
    port_capacities: tuple[int, ...] = ()
    link_capacities: dict[tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        qpus = len(self.capacities)
        if qpus < 2 or any(value < 1 for value in self.capacities):
            raise ValueError("at least two positive-capacity QPUs are required")
        if len(self.local_edges) != qpus:
            raise ValueError("one local coupling map is required per QPU")
        if self.port_capacities and len(self.port_capacities) != qpus:
            raise ValueError("one port capacity is required per QPU")

    @property
    def num_qpus(self) -> int:
        return len(self.capacities)

    def is_feasible(self, assignment: Sequence[int]) -> bool:
        counts = [0] * self.num_qpus
        for qpu in assignment:
            if not 0 <= int(qpu) < self.num_qpus:
                return False
            counts[int(qpu)] += 1
        return all(count <= cap for count, cap in zip(counts, self.capacities))

    def shortest_path(self, source: int, target: int) -> tuple[int, ...]:
        if source == target:
            return (source,)
        graph = {q: [] for q in range(self.num_qpus)}
        for left, right in self.network_edges:
            graph[left].append(right)
            graph[right].append(left)
        queue = [(source, (source,))]
        seen = {source}
        for node, path in queue:
            for nxt in sorted(graph[node]):
                if nxt == target:
                    return path + (nxt,)
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, path + (nxt,)))
        raise ValueError(f"QPU network is disconnected: {source}->{target}")


@dataclass(frozen=True)
class AssignmentMetrics:
    assignment: tuple[int, ...]
    objective: float
    remote_rounds: int
    epr_hops: int
    local_added_2q: int
    remote_gate_count: int
    details: dict[str, float] = field(default_factory=dict)


def hamming_distance(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) != len(right):
        raise ValueError("assignments must have equal length")
    return sum(int(a) != int(b) for a, b in zip(left, right))


def line_hardware(num_qpus: int, capacity: int) -> HardwareModel:
    """Build the compact line-topology hardware used by examples."""
    network = tuple((q, q + 1) for q in range(num_qpus - 1))
    local = tuple(tuple((q, q + 1) for q in range(capacity - 1)) for _ in range(num_qpus))
    return HardwareModel(
        capacities=(capacity,) * num_qpus,
        network_edges=network,
        local_edges=local,
        port_capacities=(1,) * num_qpus,
        link_capacities={edge: 1 for edge in network},
    )


def ring_hardware(num_qpus: int, capacity: int) -> HardwareModel:
    """Build the line-coupled QPUs and ring network used by experiments 74/75."""
    edges = {
        tuple(sorted((index, (index + 1) % num_qpus)))
        for index in range(num_qpus)
    } if num_qpus >= 2 else set()
    network = tuple(sorted(edges))
    local = tuple(
        tuple((q, q + 1) for q in range(capacity - 1))
        for _ in range(num_qpus)
    )
    return HardwareModel(
        capacities=(capacity,) * num_qpus,
        network_edges=network,
        local_edges=local,
        port_capacities=(1,) * num_qpus,
        link_capacities={edge: 1 for edge in network},
    )


def pairwise_layers(num_qubits: int, edges: Iterable[tuple[int, int]]) -> LayeredCircuit:
    """Greedily place arbitrary interactions into conflict-free layers."""
    layers: list[list[tuple[int, int]]] = []
    for left, right in edges:
        for layer in layers:
            used = {q for edge in layer for q in edge}
            if left not in used and right not in used:
                layer.append((left, right))
                break
        else:
            layers.append([(left, right)])
    return LayeredCircuit(num_qubits, tuple(tuple(layer) for layer in layers))


def generate_clustered_circuit(num_qubits: int, depth: int, seed: int = 0) -> LayeredCircuit:
    rng = Random(seed)
    edges: list[tuple[int, int]] = []
    split = max(1, num_qubits // 2)
    for _ in range(depth):
        cluster = range(0, split) if rng.random() < 0.5 else range(split, num_qubits)
        choices = list(combinations(cluster, 2))
        if choices:
            edges.append(rng.choice(choices))
    return pairwise_layers(num_qubits, edges)


def qiskit_to_layered(circuit) -> LayeredCircuit:
    """Extract two-qubit interactions from a Qiskit circuit."""
    edges = []
    for item in circuit.data:
        if len(item.qubits) == 2:
            edges.append(tuple(circuit.find_bit(bit).index for bit in item.qubits))
    return pairwise_layers(circuit.num_qubits, edges)
