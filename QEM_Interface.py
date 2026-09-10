"""Convert external 16-qubit circuits to model-ready PK data and run QEM."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from qiskit import qasm2

from UC_QAOA16 import (
    N_PHYSICAL, QPU_MAP, SINGLE_Z_NAMES, build_dynamic_tp, exact_logical_expectations,
    gate_records, make_uc_physical_noise, noise_model, observable_definitions,
    run_dynamic_once, UC_APPLICATION_ALL_BITS,
)
from DQC_Noise_Model import stable_seed
import Hierarchical_GNN_QEM as qm


REQUIRED_MODEL_FIELDS = (
    "sample_id", "single_z_names", "logical_ideal", "distributed_noisy",
    "gate_records", "communication_events", "qpu_map", "noise",
    "circuit_features",
)


def raw_distribution(sample):
    """Recover the complete 16-bit noisy distribution used by experiments 74/75."""
    probability = np.full(1 << 16, 1e-12, float)
    for key, count in sample["noisy_counts"].items():
        text = str(key).split()[0].zfill(16)[::-1]
        index = sum(int(value) << qubit for qubit, value in enumerate(text))
        probability[index] += int(count)
    return probability / probability.sum()


def calibrate_joint(probability, mitigated_z, iterations=60):
    """Match the noisy joint distribution marginals to GNN-mitigated Z values."""
    result = np.asarray(probability, float).copy()
    target = np.clip((1 - np.asarray(mitigated_z)) / 2, .01, .99)
    for _ in range(iterations):
        for qubit in range(16):
            mask = UC_APPLICATION_ALL_BITS[:, qubit] == 1
            current = float(result[mask].sum())
            result[mask] *= target[qubit] / max(current, 1e-12)
            result[~mask] *= (1 - target[qubit]) / max(1 - current, 1e-12)
            result /= result.sum()
    return result


def circuit_to_sample(logical_circuit, sample_id, shots=2048, noise=None,
                      logical_ideal=None, generator_info=None):
    """Build a model-ready sample from an external 16-qubit logical circuit.

    The current trained models use the fixed four-QPU mapping ``q // 4``.
    Dynamic-TP conversion is performed here so that gate and communication
    metadata exactly match the training representation.
    """
    if logical_circuit.num_qubits != 16:
        raise ValueError(f"Expected a 16-qubit circuit, got {logical_circuit.num_qubits}")
    sample_id = str(sample_id)
    noise = make_uc_physical_noise(sample_id) if noise is None else noise
    distributed, events, operations = build_dynamic_tp(logical_circuit, noise)
    definitions = observable_definitions()
    if logical_ideal is None:
        logical_ideal = exact_logical_expectations(logical_circuit, definitions)
    seed = stable_seed(f"external_noisy_{sample_id}", 37001)
    noisy, counts = run_dynamic_once(
        distributed, definitions, int(shots), seed, noise_model(noise, operations))
    records = gate_records(logical_circuit, noise)
    two_qubit = sum(r["num_qubits"] == 2 for r in records)
    distances = [float(e.get("distance_km", 0.0)) for e in events]
    qasm = qasm2.dumps(logical_circuit)
    features = {
        "num_data_qubits": 16, "num_physical_qubits": N_PHYSICAL,
        "num_qpus": 4, "qubits_per_qpu": 4,
        "logical_depth": int(logical_circuit.depth()),
        "logical_gate_count": int(logical_circuit.size()),
        "distributed_depth": int(distributed.depth()),
        "distributed_gate_count": int(distributed.size()),
        "one_qubit_gate_count": sum(r["num_qubits"] == 1 for r in records),
        "two_qubit_gate_count": two_qubit,
        "remote_gate_count": len(events),
        "local_gate_count": len(records) - len(events),
        "communication_ratio_all_gates": len(events) / max(1, len(records)),
        "communication_ratio_two_qubit": len(events) / max(1, two_qubit),
        "communication_distance_total_km": float(sum(distances)),
        "communication_distance_mean_km": float(np.mean(distances)) if distances else 0.0,
        "gate_type_counts": {str(k): int(v) for k, v in logical_circuit.count_ops().items()},
    }
    return {
        "schema_version": "qaoa16_external_single_z_qem_interface_v1",
        "sample_id": sample_id, "split": "test", "benchmark": "external-qaoa16",
        "generator_info": dict(generator_info or {}), "shots": int(shots),
        "observable_definitions": definitions, "observable_names": list(definitions),
        "single_z_names": list(SINGLE_Z_NAMES),
        "logical_ideal": {k: float(v) for k, v in logical_ideal.items()},
        "distributed_noisy": noisy, "mitigated": dict(noisy),
        "logical_circuit": logical_circuit, "distributed_circuit": distributed,
        "logical_qasm": qasm, "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
        "qpu_map": dict(QPU_MAP), "noise": noise, "gate_records": records,
        "communication_events": events, "distributed_operation_records": operations,
        "circuit_features": features, "noisy_counts": counts,
    }


def validate_sample(sample):
    missing = [key for key in REQUIRED_MODEL_FIELDS if key not in sample]
    if missing:
        raise ValueError(f"Sample {sample.get('sample_id', '<unknown>')} missing fields: {missing}")
    if tuple(sample["single_z_names"]) != tuple(SINGLE_Z_NAMES):
        raise ValueError(
            f"Expected observables {SINGLE_Z_NAMES}, got {sample['single_z_names']}")
    return sample


def save_pk_shards(samples, root, split="test", samples_per_pk=100):
    """Save complete samples in the same shard layout used by notebook 29."""
    samples = [validate_sample(s) for s in samples]
    target = Path(root) / split
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for step, start in enumerate(range(0, len(samples), int(samples_per_pk))):
        rows = samples[start:start + int(samples_per_pk)]
        path = target / f"step_{step}.pk"
        temporary = path.with_suffix(".pk.tmp")
        with temporary.open("wb") as f:
            pickle.dump({"schema_version": "qaoa16_external_single_z_qem_interface_v1",
                         "sample_count": len(rows), "samples": rows},
                        f, pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        paths.append(path)
    return paths


def load_pk_samples(root, split="test"):
    rows = []
    paths = sorted((Path(root) / split).glob("step_*.pk"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not paths:
        raise FileNotFoundError(f"No PK shards under {Path(root) / split}")
    for path in paths:
        with path.open("rb") as f:
            payload = pickle.load(f)
        rows.extend(validate_sample(s) for s in payload["samples"])
    return rows


def _load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("num_outputs") != len(SINGLE_Z_NAMES):
        raise ValueError(f"Wrong output count in {path}: {checkpoint.get('num_outputs')}")
    if tuple(checkpoint.get("observable_names", ())) != tuple(SINGLE_Z_NAMES):
        raise ValueError(f"Wrong observable order in {path}")
    return checkpoint


def load_gnn_models(model_root, device=None):
    """Load the three metadata-complete single-Z GNN checkpoints."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = Path(model_root)
    # qaoa16_models keeps the task output size as module state. Set it before
    # constructing models so a notebook kernel that previously ran ZZ is safe.
    qm.NUM_OUTPUTS = len(SINGLE_Z_NAMES)
    qm.names = lambda sample: tuple(sample["single_z_names"])
    specs = {
        "global_gnn": qm.QAOA16GNN(hidden=160, steps=6, dropout=.12),
        "hierarchical_gnn": qm.QAOA16HierarchicalGNN(
            hidden=160, local_steps=6, qpu_steps=4, dropout=.12),
        "dual_branch_hierarchical_gnn": qm.QAOA16DualBranchHierarchicalGNN(
            hidden=160, local_steps=6, qpu_steps=4, dropout=.12),
    }
    models = {}
    for name, model in specs.items():
        checkpoint = _load_checkpoint(root / f"{name}.pt")
        model.load_state_dict(checkpoint["state_dict"])
        models[name] = model.to(device).eval()
    return models, device


def load_edge_aware_model(checkpoint_path, device=None):
    """Load the edge-aware hierarchical GNN used by experiments 74/75."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    qm.NUM_OUTPUTS = 16
    qm.names = lambda sample: tuple(sample["single_z_names"])
    model = qm.QAOA16EdgeAwareHierarchicalGNN(
        hidden=160,
        local_steps=6,
        qpu_steps=4,
        dropout=.12,
    ).to(device)
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, device


def mitigate_uc_application_sample(sample, model, device):
    graph = qm.build_dual_graph(sample)
    graph["qpu"].batch = torch.zeros(graph["qpu"].num_nodes, dtype=torch.long)
    graph = graph.to(device)
    with torch.no_grad():
        return np.clip(model(graph).detach().cpu().numpy().ravel(), -1, 1)


def mitigate_samples(samples, model_root, device=None, batch_size=24):
    """Apply the three trained GNNs and return long-form test predictions."""
    samples = [validate_sample(s) for s in samples]
    models, device = load_gnn_models(model_root, device)
    names = tuple(SINGLE_Z_NAMES)
    frames = []
    for method, model in models.items():
        graphs = (qm.build_graphs(samples, "logical_global") if method == "global_gnn"
                  else qm.build_dual_graphs(samples))
        frame = qm.predict_graph(model, graphs, names, batch_size=batch_size)
        frame["method"] = method
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    ideal = {(s["sample_id"], n): s["logical_ideal"][n] for s in samples for n in names}
    noisy = {(s["sample_id"], n): s["distributed_noisy"][n] for s in samples for n in names}
    keys = list(zip(result["sample_id"], result["observable"]))
    result["logical_ideal"] = [ideal[k] for k in keys]
    result["distributed_noisy"] = [noisy[k] for k in keys]
    return result[["sample_id", "observable", "method", "logical_ideal",
                   "distributed_noisy", "mitigated"]]
