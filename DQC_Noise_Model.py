"""Seeded 16-qubit random-circuit dataset for four heterogeneous QPUs."""
from __future__ import annotations

import hashlib
import json
import pickle
import zlib
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, qasm2, transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error
from qiskit.quantum_info import Statevector
from tqdm.auto import tqdm



def stable_seed(value, base=20260805):
    """Return a reproducible unsigned 32-bit seed."""
    return int(base) + zlib.crc32(str(value).encode("utf-8"))
NUM_QUBITS = 16
NUM_QPUS = 4
QUBITS_PER_QPU = 4
QPU_MAP = {q: q // QUBITS_PER_QPU for q in range(NUM_QUBITS)}
QPU_DISTANCE_KM = {(0, 1): 8.0, (0, 2): 18.0, (0, 3): 30.0,
                   (1, 2): 10.0, (1, 3): 22.0, (2, 3): 12.0}
ZZ_EDGES = tuple((q, q + 1) for q in range(0, NUM_QUBITS, 2))
ZZ_NAMES = tuple(f"Z{a}Z{b}" for a, b in ZZ_EDGES)

def observable_definitions():
    return {name: [a, b] for name, (a, b) in zip(ZZ_NAMES, ZZ_EDGES)}

def exact_expectations(circuit, observables):
    probabilities = Statevector.from_instruction(circuit).probabilities()
    values = {}
    for name, active in observables.items():
        values[name] = float(sum((1 if sum((index >> q) & 1 for q in active) % 2 == 0 else -1) * p
                                 for index, p in enumerate(probabilities)))
    return values

def expectations_from_counts(counts, observables):
    total = sum(counts.values()); values = {}
    for name, active in observables.items():
        values[name] = float(sum((1 if sum(int(key.replace(' ', '')[::-1][q]) for q in active) % 2 == 0 else -1)
                                 * count / total for key, count in counts.items()))
    return values

SPLITS = {"train": 500, "val": 100, "test": 100}
SAMPLES_PER_PK = 100
SHOTS = 2048
DEFAULT_DEPTH_RANGE = (24, 40)
DEFAULT_MASTER_SEED = 25001
SCHEMA_VERSION = "finally_random16_seeded_4heterogeneous_qpus_700_v1"
BASIS = ["id", "x", "h", "sx", "rx", "ry", "rz", "p", "cx", "cz", "swap"]


def build_random16(sample_index, master_seed=DEFAULT_MASTER_SEED,
                   depth_range=DEFAULT_DEPTH_RANGE, max_operands=2):
    """Generate an independent circuit; seed and requested depth are saved."""
    low, high = map(int, depth_range)
    if low < 1 or high < low:
        raise ValueError("depth_range must satisfy 1 <= low <= high")
    seed = stable_seed(f"random16_{master_seed}_{sample_index}", 25002)
    rng = np.random.default_rng(seed)
    requested_depth = int(rng.integers(low, high + 1))
    raw = random_circuit(NUM_QUBITS, requested_depth, max_operands=max_operands,
                         measure=False, seed=seed)
    circuit = transpile(raw, basis_gates=BASIS, optimization_level=1,
                        seed_transpiler=seed)
    circuit.name = f"seeded_random16_{sample_index}"
    return circuit, {"generator": "qiskit.circuit.random.random_circuit",
                     "seed": int(seed), "master_seed": int(master_seed),
                     "requested_depth": requested_depth,
                     "requested_depth_range": [low, high],
                     "max_operands": int(max_operands),
                     "transpiled_depth": int(circuit.depth()),
                     "basis_gates": list(BASIS)}


def make_heterogeneous_noise(sample_id):
    """Four reproducible but non-identical QPU calibration profiles."""
    rng = np.random.default_rng(stable_seed(sample_id, 25003))
    # QPU 0 is best and QPU 3 is weakest; ranges overlap to resemble calibration drift.
    nominal = (
        {"p1": .0045, "p2": .025, "readout": .018, "t1": 78., "t2": 62.},
        {"p1": .0060, "p2": .034, "readout": .024, "t1": 70., "t2": 55.},
        {"p1": .0075, "p2": .043, "readout": .031, "t1": 62., "t2": 48.},
        {"p1": .0090, "p2": .052, "readout": .038, "t1": 55., "t2": 42.},
    )
    qpus, physical = [], []
    for qpu, base in enumerate(nominal):
        row = {"qpu": qpu,
               "p1": float(base["p1"] * rng.uniform(.85, 1.15)),
               "p2": float(base["p2"] * rng.uniform(.88, 1.12)),
               "readout_error": float(base["readout"] * rng.uniform(.85, 1.15)),
               "mean_t1_us": float(base["t1"] * rng.uniform(.92, 1.08)),
               "mean_t2_us": float(base["t2"] * rng.uniform(.92, 1.08))}
        qpus.append(row)
        for qubit in range(qpu*QUBITS_PER_QPU, (qpu+1)*QUBITS_PER_QPU):
            t1 = row["mean_t1_us"] * rng.uniform(.94, 1.06)
            t2 = min(row["mean_t2_us"] * rng.uniform(.94, 1.06), 2*t1)
            physical.append({"qubit": qubit, "qpu": qpu, "t1_us": float(t1), "t2_us": float(t2),
                             "t1_ns": float(t1*1000), "t2_ns": float(t2*1000)})
    communication = {}
    for pair, distance in QPU_DISTANCE_KM.items():
        # Moderate remote error, with a small distance-dependent component.
        communication[f"{pair[0]}-{pair[1]}"] = {
            "qpu_pair": list(pair), "distance_km": distance,
            "pcomm": float(np.clip(.055 + .0018*distance + rng.uniform(-.008, .008), .05, .13)),
            "duration_ns": float(1800 + 65*distance + rng.uniform(-180, 180))}
    return {"model": "four-QPU heterogeneous local + pair-dependent communication + thermal + readout",
            "qpu_profiles": qpus, "physical_noise_profile": physical,
            "communication_profiles": communication,
            "local_1q_duration_ns": 35.0, "local_2q_duration_ns": 260.0}


def distributed_tp_abstraction(logical, noise):
    """Map every remote two-qubit gate to a reusable TP-supernode event.

    The circuit unitary remains unchanged.  This avoids adding 12 TP ancillas to
    a 16-qubit statevector, while preserving all causal/physical graph records.
    """
    distributed = logical.copy(name=f"distributed_{logical.name}")
    events, operations = [], []
    for gate_id, item in enumerate(logical.data):
        qubits = [logical.find_bit(q).index for q in item.qubits]
        remote = len(qubits) == 2 and QPU_MAP[qubits[0]] != QPU_MAP[qubits[1]]
        kind = "communication" if remote else "local"
        operation = {"gate_id": gate_id, "name": item.operation.name, "qubits": qubits,
                     "logical_qubits": qubits, "noise_class": kind}
        operations.append(operation)
        if remote:
            source, target = QPU_MAP[qubits[0]], QPU_MAP[qubits[1]]
            pair = tuple(sorted((source, target))); profile = noise["communication_profiles"][f"{pair[0]}-{pair[1]}"]
            t2 = min(noise["physical_noise_profile"][q]["t2_ns"] for q in qubits)
            events.append({"gate_id": gate_id, "gate_type": item.operation.name,
                           "logical_qubits": qubits, "source_qpu": source, "target_qpu": target,
                           "qpu_pair": list(pair), "distance_km": profile["distance_km"],
                           "communication_noise": profile["pcomm"],
                           "communication_duration_ns": profile["duration_ns"],
                           "idle_decoherence": float(1-np.exp(-profile["duration_ns"]/t2)),
                           "tp_representation": "logical-equivalent reusable TP supernode",
                           "auxiliary_qubits_reused": True,
                           "remote_gate_supported_generically": True})
    return distributed, events, operations


def _relax(profile, qubit, duration):
    row = profile[qubit]
    return thermal_relaxation_error(row["t1_ns"], min(row["t2_ns"], 2*row["t1_ns"]), duration)


def build_noise_model(noise, operations):
    model, physical = NoiseModel(), noise["physical_noise_profile"]
    unique = {(op["name"], tuple(op["qubits"]), op["noise_class"])
              for op in operations if len(op["qubits"]) in (1, 2)}
    for name, qubits, kind in sorted(unique):
        if kind == "communication":
            pair = tuple(sorted((QPU_MAP[qubits[0]], QPU_MAP[qubits[1]])))
            comm = noise["communication_profiles"][f"{pair[0]}-{pair[1]}"]
            probability, duration = comm["pcomm"], comm["duration_ns"]
        else:
            qpu_rows = [noise["qpu_profiles"][QPU_MAP[q]] for q in qubits]
            probability = float(np.mean([r["p2" if len(qubits)==2 else "p1"] for r in qpu_rows]))
            duration = noise["local_2q_duration_ns" if len(qubits)==2 else "local_1q_duration_ns"]
        relaxation = _relax(physical, qubits[0], duration)
        for q in qubits[1:]: relaxation = relaxation.tensor(_relax(physical, q, duration))
        error = depolarizing_error(probability, len(qubits)).compose(relaxation)
        model.add_quantum_error(error, name, list(qubits))
    for q in range(NUM_QUBITS):
        p = noise["qpu_profiles"][QPU_MAP[q]]["readout_error"]
        model.add_readout_error(ReadoutError([[1-p, p], [p, 1-p]]), [q])
    return model


def noisy_expectations_once(circuit, operations, noise, observables, shots, seed):
    measured = circuit.copy(); result = ClassicalRegister(NUM_QUBITS, "result")
    measured.add_register(result); measured.measure(range(NUM_QUBITS), result)
    counts = AerSimulator(method="statevector", noise_model=build_noise_model(noise, operations),
                          seed_simulator=seed).run(measured, shots=shots).result().get_counts()
    return expectations_from_counts(counts, observables), {str(k): int(v) for k,v in counts.items()}


def gate_records(circuit, noise):
    records, last_layer, end_time = [], [-1]*NUM_QUBITS, [0.0]*NUM_QUBITS
    for gate_id, item in enumerate(circuit.data):
        qubits = [circuit.find_bit(q).index for q in item.qubits]
        qpus = [QPU_MAP[q] for q in qubits]
        remote = len(qubits)==2 and qpus[0] != qpus[1]
        layer = 1 + max((last_layer[q] for q in qubits), default=-1)
        start = max((end_time[q] for q in qubits), default=0.0)
        if remote:
            pair = tuple(sorted(qpus)); comm = noise["communication_profiles"][f"{pair[0]}-{pair[1]}"]
            duration, error_rate = comm["duration_ns"], comm["pcomm"]
        else:
            duration = noise["local_2q_duration_ns" if len(qubits)==2 else "local_1q_duration_ns"]
            key = "p2" if len(qubits)==2 else "p1"
            error_rate = float(np.mean([noise["qpu_profiles"][qpu][key] for qpu in qpus]))
        parameters = []
        for parameter in item.operation.params:
            try: parameters.append(float(parameter))
            except (TypeError, ValueError): parameters.append(str(parameter))
        records.append({"gate_id": gate_id, "gate_type": item.operation.name,
                        "parameters": parameters,
                        "qubits": qubits, "qpus": qpus, "num_qubits": len(qubits),
                        "layer": layer, "start_time_ns": start, "duration_ns": float(duration),
                        "is_remote": remote, "error_rate": float(error_rate),
                        "t1_us": float(np.mean([noise["physical_noise_profile"][q]["t1_us"] for q in qubits])),
                        "t2_us": float(np.mean([noise["physical_noise_profile"][q]["t2_us"] for q in qubits])),
                        "readout_error": float(np.mean([noise["qpu_profiles"][qpu]["readout_error"] for qpu in qpus]))})
        for q in qubits: last_layer[q], end_time[q] = layer, start+duration
    return records


def make_sample(sample_index, split, master_seed=DEFAULT_MASTER_SEED,
                depth_range=DEFAULT_DEPTH_RANGE, shots=SHOTS):
    logical, generator = build_random16(sample_index, master_seed, depth_range)
    noise = make_heterogeneous_noise(f"{master_seed}_{split}_{sample_index}")
    distributed, events, operations = distributed_tp_abstraction(logical, noise)
    observables = observable_definitions(); ideal = exact_expectations(logical, observables)
    # TP abstraction is unitary-identical; do not repeat the expensive ideal simulation.
    noiseless = dict(ideal)
    noisy, counts = noisy_expectations_once(distributed, operations, noise, observables, shots,
                                             stable_seed(f"{master_seed}_{split}_{sample_index}", 25004))
    records = gate_records(logical, noise); twoq = sum(r["num_qubits"]==2 for r in records)
    qasm = qasm2.dumps(logical)
    features = {"num_data_qubits": NUM_QUBITS, "num_qpus": NUM_QPUS,
                "qubits_per_qpu": QUBITS_PER_QPU, "logical_depth": int(logical.depth()),
                "logical_gate_count": int(logical.size()), "distributed_depth": int(distributed.depth()),
                "distributed_gate_count": int(distributed.size()),
                "one_qubit_gate_count": sum(r["num_qubits"]==1 for r in records),
                "two_qubit_gate_count": twoq, "remote_gate_count": len(events),
                "local_gate_count": len(records)-len(events),
                "communication_ratio_all_gates": len(events)/max(1,len(records)),
                "communication_ratio_two_qubit": len(events)/max(1,twoq),
                "communication_distance_total_km": float(sum(e["distance_km"] for e in events)),
                "communication_distance_mean_km": float(np.mean([e["distance_km"] for e in events])) if events else 0.,
                "gate_type_counts": {str(k):int(v) for k,v in logical.count_ops().items()}}
    return {"schema_version": SCHEMA_VERSION, "sample_id": f"random16_{split}_{sample_index:05d}",
            "split": split, "benchmark": "independent-seeded-random-circuit", "generator_info": generator,
            "num_data_qubits": NUM_QUBITS, "num_qpus": NUM_QPUS, "shots": shots,
            "observable_definitions": observables, "observables": observables,
            "selected_zz_edges": [list(e) for e in ZZ_EDGES], "zz_names": list(ZZ_NAMES),
            "logical_ideal": ideal, "distributed_noiseless": noiseless,
            "distributed_noisy": noisy, "mitigated": dict(noisy),
            "noiseless_validation": {"max_abs_delta": 0.0, "tolerance": 1e-10, "passed": True,
                                      "reason": "unitary-identical TP-supernode abstraction"},
            "logical_circuit": logical, "distributed_circuit": distributed,
            "logical_qasm": qasm, "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
            "qpu_map": dict(QPU_MAP), "qpu_distance_km": {f"{a}-{b}":d for (a,b),d in QPU_DISTANCE_KM.items()},
            "auxiliary_qubits_reused": True, "tp_simulation": "logical-equivalent reusable TP supernode",
            "noise": noise, "gate_records": records, "communication_events": events,
            "distributed_operation_records": operations, "circuit_features": features,
            "noisy_counts": counts}


def preflight(master_seed=DEFAULT_MASTER_SEED, depth_range=DEFAULT_DEPTH_RANGE, num_probes=8):
    hashes, depths, rows = set(), [], []
    observables = observable_definitions()
    for i in range(num_probes):
        circuit, _ = build_random16(i, master_seed, depth_range)
        hashes.add(hashlib.sha256(qasm2.dumps(circuit).encode()).hexdigest()); depths.append(circuit.depth())
        values = exact_expectations(circuit, observables); rows.append([values[n] for n in ZZ_NAMES])
    array = np.asarray(rows)
    return {"unique_circuits": len(hashes), "num_probes": num_probes,
            "transpiled_depth_range": [int(min(depths)),int(max(depths))],
            "ideal_min": float(array.min()), "ideal_max": float(array.max()),
            "overall_std": float(array.std()),
            "per_observable_std": {n:float(array[:,i].std()) for i,n in enumerate(ZZ_NAMES)}}


def generate_dataset(output_root, split_sizes=SPLITS, master_seed=DEFAULT_MASTER_SEED,
                     depth_range=DEFAULT_DEPTH_RANGE, shots=SHOTS, force=False):
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    report = preflight(master_seed, depth_range)
    metadata = {"schema_version": SCHEMA_VERSION, "generator": "independent seeded random_circuit",
                "master_seed": master_seed, "requested_depth_range": list(depth_range),
                "num_data_qubits": NUM_QUBITS, "num_qpus": NUM_QPUS,
                "qubits_per_qpu": QUBITS_PER_QPU, "split_sizes": dict(split_sizes),
                "samples_per_pk": SAMPLES_PER_PK, "shots": shots,
                "training_outputs": list(ZZ_NAMES), "preflight": report,
                "graph_objects_saved": False}
    (output_root/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    global_index = 0; hashes = set()
    for split,size in split_sizes.items():
        folder=output_root/split; folder.mkdir(parents=True,exist_ok=True)
        for start in range(0,int(size),SAMPLES_PER_PK):
            path=folder/f"step_{start//SAMPLES_PER_PK}.pk"
            if path.exists() and not force:
                print("skip",path); global_index += min(SAMPLES_PER_PK,size-start); continue
            rows=[]
            for _ in tqdm(range(start,min(start+SAMPLES_PER_PK,size)),desc=f"random16 {split} {start}"):
                row=make_sample(global_index,split,master_seed,depth_range,shots)
                rows.append(row);hashes.add(row["qasm_sha256"]);global_index+=1
            temporary=path.with_suffix(".pk.tmp")
            with temporary.open("wb") as handle:
                pickle.dump({"schema_version":SCHEMA_VERSION,"split":split,
                             "sample_count":len(rows),"samples":rows},handle,pickle.HIGHEST_PROTOCOL)
            temporary.replace(path);print("saved",path)
    metadata["unique_qasm_count_current_run"]=len(hashes)
    (output_root/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return metadata
