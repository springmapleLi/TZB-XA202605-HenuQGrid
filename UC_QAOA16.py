"""16-qubit Unit-Commitment QAOA on four heterogeneous QPUs.

Qubit ``4*t+g`` represents generator g at time t (4 generators x 4 periods).
Each sample stores its complete UC instance, QUBO/Ising coefficients, QAOA
parameters, dynamic-TP circuit, physical noise and 25 Z-type expectations.

The 2026-909 application uses the same 16 qubits as 8 generators x 2 periods
and exposes its fixed UC instance, sparse interaction graph and noisy sample
builder below.  That block is reformatted from notebooks 74/75 and
``insert/uc_distributed.py``.
"""
from __future__ import annotations
import hashlib,json,pickle
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit,qasm2
from tqdm.auto import tqdm
import DQC_Noise_Model as hw
from DQC_Noise_Model import stable_seed
from DQC_Noise_Model import NUM_QUBITS,NUM_QPUS,QUBITS_PER_QPU,QPU_MAP,make_heterogeneous_noise,gate_records
from Dynamic_TeleGate import (N_PHYSICAL,_extend_aux_profiles,build_dynamic_tp,
 exact_logical_expectations,noise_model,require_gpu,run_dynamic_once)

SPLITS={'train':1600,'val':100,'test':100};SAMPLES_PER_PK=100;SHOTS=2048
DEFAULT_P_RANGE=(1,4);DEFAULT_MASTER_SEED=29001
NUM_GENERATORS=4;NUM_PERIODS=4
SCHEMA_VERSION='finally_uc_qaoa16_4gen4period_variable_depth_dynamic_tp_gpu_700_v1'
SINGLE_Z_NAMES=tuple(f'Z{q}' for q in range(NUM_QUBITS))
ZZ_EDGES=tuple((q,q+1) for q in range(0,NUM_QUBITS,2))
ZZ_NAMES=tuple(f'Z{a}Z{b}' for a,b in ZZ_EDGES)
GLOBAL_Z_NAME='Z_ALL_16';OBSERVABLE_NAMES=SINGLE_Z_NAMES+ZZ_NAMES+(GLOBAL_Z_NAME,)

# 2026-909 notebooks 74/75: 8 generators x 2 periods.
UC_APPLICATION_UNITS = 8
UC_APPLICATION_PERIODS = 2
UC_APPLICATION_PMAX = np.array([70, 90, 110, 130, 150, 170, 100, 120.], float)
UC_APPLICATION_STARTUP = np.array([7, 9, 12, 15, 19, 24, 10, 14.], float)
UC_APPLICATION_LINEAR = np.array([1.02, 1.08, 1.16, 1.27, 1.40, 1.55, 1.12, 1.23], float)
UC_APPLICATION_QUADRATIC = np.array([.005, .006, .007, .008, .010, .012, .0065, .0075], float)
UC_APPLICATION_SWITCH = np.array([9, 10, 12, 14, 17, 20, 11, 13.], float)
UC_APPLICATION_DEMANDS = np.array([500., 620.])
UC_APPLICATION_PENALTY = 4.0
UC_APPLICATION_ALL_BITS = np.array(
    [[(index >> qubit) & 1 for qubit in range(NUM_QUBITS)] for index in range(1 << NUM_QUBITS)],
    int,
)


def evaluate_uc_application(bits):
    on = np.asarray(bits, int).reshape(UC_APPLICATION_PERIODS, UC_APPLICATION_UNITS)
    power = on * UC_APPLICATION_PMAX
    supply = power.sum(1)
    deviation = np.abs(UC_APPLICATION_DEMANDS - supply)
    running = np.sum(
        on * UC_APPLICATION_STARTUP
        + UC_APPLICATION_LINEAR * power
        + UC_APPLICATION_QUADRATIC * power**2
    )
    switching = np.sum(UC_APPLICATION_SWITCH * np.abs(on[1:] - on[:-1]))
    return float(running + switching), float(deviation.sum()), power


def uc_application_objective(bits):
    cost, _, power = evaluate_uc_application(bits)
    return cost + UC_APPLICATION_PENALTY * float(
        np.sum((power.sum(1) - UC_APPLICATION_DEMANDS) ** 2)
    )


UC_APPLICATION_REFERENCE_BITS = UC_APPLICATION_ALL_BITS[
    int(np.argmin([uc_application_objective(bits) for bits in UC_APPLICATION_ALL_BITS]))
]
UC_APPLICATION_REFERENCE_TEXT = "".join(map(str, UC_APPLICATION_REFERENCE_BITS))
UC_APPLICATION_REFERENCE_COST = evaluate_uc_application(UC_APPLICATION_REFERENCE_BITS)[0]

_application_run_cost = (
    UC_APPLICATION_STARTUP
    + UC_APPLICATION_LINEAR * UC_APPLICATION_PMAX
    + UC_APPLICATION_QUADRATIC * UC_APPLICATION_PMAX**2
)
UC_APPLICATION_QUBO_LINEAR = np.tile(_application_run_cost, UC_APPLICATION_PERIODS)
UC_APPLICATION_QUBO_PAIR = {}
for _period in range(UC_APPLICATION_PERIODS):
    for _unit in range(UC_APPLICATION_UNITS):
        _index = _period * UC_APPLICATION_UNITS + _unit
        UC_APPLICATION_QUBO_LINEAR[_index] += UC_APPLICATION_PENALTY * (
            UC_APPLICATION_PMAX[_unit] ** 2
            - 2 * UC_APPLICATION_DEMANDS[_period] * UC_APPLICATION_PMAX[_unit]
        )
    for _unit in range(UC_APPLICATION_UNITS):
        for _other in range(_unit + 1, UC_APPLICATION_UNITS):
            UC_APPLICATION_QUBO_PAIR[
                (_period * UC_APPLICATION_UNITS + _unit,
                 _period * UC_APPLICATION_UNITS + _other)
            ] = 2 * UC_APPLICATION_PENALTY * UC_APPLICATION_PMAX[_unit] * UC_APPLICATION_PMAX[_other]
for _period in range(1, UC_APPLICATION_PERIODS):
    for _unit in range(UC_APPLICATION_UNITS):
        _left = (_period - 1) * UC_APPLICATION_UNITS + _unit
        _right = _period * UC_APPLICATION_UNITS + _unit
        UC_APPLICATION_QUBO_LINEAR[_left] += UC_APPLICATION_SWITCH[_unit]
        UC_APPLICATION_QUBO_LINEAR[_right] += UC_APPLICATION_SWITCH[_unit]
        UC_APPLICATION_QUBO_PAIR[(_left, _right)] = (
            UC_APPLICATION_QUBO_PAIR.get((_left, _right), 0.)
            - 2 * UC_APPLICATION_SWITCH[_unit]
        )

UC_APPLICATION_FIELDS = -.5 * UC_APPLICATION_QUBO_LINEAR.copy()
UC_APPLICATION_COUPLINGS = {}
for (_left, _right), _value in UC_APPLICATION_QUBO_PAIR.items():
    UC_APPLICATION_FIELDS[_left] -= _value / 4
    UC_APPLICATION_FIELDS[_right] -= _value / 4
    UC_APPLICATION_COUPLINGS[(_left, _right)] = _value / 4
_application_scale = max(
    float(np.max(np.abs(UC_APPLICATION_FIELDS))),
    max(abs(value) for value in UC_APPLICATION_COUPLINGS.values()),
)
UC_APPLICATION_FIELDS /= _application_scale
_application_temporal = {
    (unit, UC_APPLICATION_UNITS + unit)
    for unit in range(UC_APPLICATION_UNITS)
}
_application_selected = [
    (edge, UC_APPLICATION_COUPLINGS[edge])
    for edge in _application_temporal
]
_application_selected += sorted(
    [(edge, value) for edge, value in UC_APPLICATION_COUPLINGS.items()
     if edge not in _application_temporal],
    key=lambda row: abs(row[1]),
    reverse=True,
)[:16]
UC_APPLICATION_EDGES = tuple(
    (left, right, float(value / _application_scale))
    for (left, right), value in _application_selected
)
UC_APPLICATION_INITIAL_ANGLES = 2 * np.arcsin(
    np.sqrt(np.where(UC_APPLICATION_REFERENCE_BITS == 1, .98, .02))
)


def build_uc_application_qaoa(gamma, beta, edges=UC_APPLICATION_EDGES,
                              fields=UC_APPLICATION_FIELDS, initial_angles=None):
    qc = QuantumCircuit(NUM_QUBITS)
    if initial_angles is None:
        qc.h(range(NUM_QUBITS))
    else:
        for qubit, angle in enumerate(initial_angles):
            qc.ry(float(angle), qubit)
    for gamma_value, beta_value in zip(gamma, beta):
        for qubit, field in enumerate(fields):
            qc.rz(2 * float(gamma_value * field), qubit)
        for left, right, coupling in edges:
            qc.cx(left, right)
            qc.rz(2 * float(gamma_value * coupling), right)
            qc.cx(left, right)
        for qubit in range(NUM_QUBITS):
            qc.rx(2 * float(beta_value), qubit)
    return qc


def make_uc_application_sample(index, gamma, beta, edges=UC_APPLICATION_EDGES,
                               fields=UC_APPLICATION_FIELDS, shots=2048,
                               noise_id=None, qpu_map=None, noise_scale=1.0,
                               initial_angles=UC_APPLICATION_INITIAL_ANGLES):
    logical = build_uc_application_qaoa(gamma, beta, edges, fields, initial_angles)
    sample_id = f"uc_qaoa16_application_{index:04d}"
    noise = hw.make_heterogeneous_noise(noise_id or sample_id)
    scale = float(noise_scale)
    for profile in noise.get("qpu_profiles", []):
        for key in ("p1", "p2", "readout_error"):
            profile[key] = float(np.clip(profile.get(key, 0.) * scale, 0., .95))
        for key in ("t1_us", "t2_us"):
            profile[key] = float(profile.get(key, 1.) / max(scale, 1e-6))
    for profile in noise.get("communication_profiles", {}).values():
        profile["pcomm"] = float(np.clip(profile.get("pcomm", 0.) * scale, 0., .95))
    noise["pcomm"] = float(np.clip(noise.get("pcomm", 0.) * scale, 0., .95))
    placement = dict(qpu_map or QPU_MAP)
    old_map = hw.QPU_MAP
    hw.QPU_MAP = placement
    distributed, events, operations = hw.distributed_tp_abstraction(logical, noise)
    observables = {f"Z{qubit}": [qubit] for qubit in range(NUM_QUBITS)}
    ideal = hw.exact_expectations(logical, observables)
    try:
        noisy, counts = hw.noisy_expectations_once(
            distributed,
            operations,
            noise,
            observables,
            shots,
            hw.stable_seed(sample_id, 25004),
        )
        records = hw.gate_records(logical, noise)
    finally:
        hw.QPU_MAP = old_map
    two_qubit = sum(record["num_qubits"] == 2 for record in records)
    features = {
        "num_data_qubits": NUM_QUBITS,
        "num_qpus": NUM_QPUS,
        "qubits_per_qpu": QUBITS_PER_QPU,
        "logical_depth": int(logical.depth()),
        "logical_gate_count": int(logical.size()),
        "distributed_depth": int(distributed.depth()),
        "distributed_gate_count": int(distributed.size()),
        "one_qubit_gate_count": sum(record["num_qubits"] == 1 for record in records),
        "two_qubit_gate_count": two_qubit,
        "remote_gate_count": len(events),
        "local_gate_count": len(records) - len(events),
        "communication_ratio_all_gates": len(events) / max(1, len(records)),
        "communication_ratio_two_qubit": len(events) / max(1, two_qubit),
        "communication_distance_total_km": float(sum(event["distance_km"] for event in events)),
        "communication_distance_mean_km": float(
            np.mean([event["distance_km"] for event in events])
        ) if events else 0.,
    }
    return {
        "sample_id": sample_id,
        "logical_circuit": logical,
        "distributed_circuit": distributed,
        "gate_records": records,
        "communication_events": events,
        "distributed_operation_records": operations,
        "circuit_features": features,
        "noise": noise,
        "single_z_names": list(observables),
        "logical_ideal": ideal,
        "distributed_noisy": noisy,
        "noisy_counts": counts,
        "qpu_map": placement,
        "num_qpus": NUM_QPUS,
        "num_data_qubits": NUM_QUBITS,
        "shots": shots,
        "generator_info": {
            "gamma": list(gamma),
            "beta": list(beta),
            "uc_edges": [list(edge) for edge in edges],
            "partitioner": "teleportation_residual_pilot.balanced_greedy_initial+greedy_correction_search",
        },
    }

def observable_definitions():
    out={name:[q] for q,name in enumerate(SINGLE_Z_NAMES)}
    out.update({name:[a,b] for name,(a,b) in zip(ZZ_NAMES,ZZ_EDGES)})
    out[GLOBAL_Z_NAME]=list(range(NUM_QUBITS));return out

def make_uc_physical_noise(sample_id):
    """Full dynamic-TP noise with small complete-event communication error."""
    noise=make_heterogeneous_noise(sample_id)
    internal_operations=13
    for profile in noise['communication_profiles'].values():
        distance=float(profile['distance_km'])
        event_p=float(np.clip(.022+.0011*distance,.025,.055))
        event_duration=float(850.+28.*distance)
        profile['pcomm']=event_p
        profile['duration_ns']=event_duration
        profile['noise_semantics']='complete TP event'
    noise=_extend_aux_profiles(noise)
    event_ps=[x['pcomm'] for x in noise['communication_profiles'].values()]
    event_durations=[x['duration_ns'] for x in noise['communication_profiles'].values()]
    per_instruction=[1-(1-p)**(1/internal_operations) for p in event_ps]
    noise['tp_internal_operations_per_event']=internal_operations
    noise['tp_event_pcomm_mean']=float(np.mean(event_ps))
    noise['tp_event_duration_ns_mean']=float(np.mean(event_durations))
    noise['effective_gpu_simulation_pcomm']=float(np.mean(per_instruction))
    noise['effective_gpu_simulation_communication_duration_ns']=float(np.mean(event_durations)/internal_operations)
    noise['communication_noise_semantics']='small complete-TP event error distributed across internal operations'
    return noise
def _q(g,t):return NUM_GENERATORS*int(t)+int(g)

def _uc_instance(rng):
    capacity=np.sort(rng.uniform(55.,125.,NUM_GENERATORS))[::-1]
    marginal=rng.uniform(12.,38.,NUM_GENERATORS);no_load=rng.uniform(4.,14.,NUM_GENERATORS)
    startup=rng.uniform(10.,32.,NUM_GENERATORS);total=float(capacity.sum())
    base=rng.uniform(.46,.68)*total
    shape=np.asarray([.88,1.02,1.12,.94])*rng.uniform(.96,1.04,NUM_PERIODS)
    demand=np.clip(base*shape,.38*total,.78*total)
    availability=np.ones((NUM_PERIODS,NUM_GENERATORS),dtype=int)
    if rng.random()<.40:
        t=int(rng.integers(NUM_PERIODS));g=int(rng.integers(NUM_GENERATORS))
        if capacity.sum()-capacity[g]>=1.08*demand[t]:availability[t,g]=0
    return {'capacity_mw':capacity,'marginal_cost':marginal,'no_load_cost':no_load,
            'startup_cost':startup,'demand_mw':demand,'availability':availability}

def _add_pair(pair,i,j,value):
    key=tuple(sorted((int(i),int(j))));pair[key]=pair.get(key,0.)+float(value)

def _uc_qubo(instance,rng):
    cap=instance['capacity_mw'];demand=instance['demand_mw'];marginal=instance['marginal_cost']
    no_load=instance['no_load_cost'];startup=instance['startup_cost'];available=instance['availability']
    linear=np.zeros(NUM_QUBITS);pair={};lam=float(rng.uniform(.018,.034));outage=float(rng.uniform(90.,140.))
    for t in range(NUM_PERIODS):
        active=[g for g in range(NUM_GENERATORS) if available[t,g]]
        for g in range(NUM_GENERATORS):
            i=_q(g,t);linear[i]+=no_load[g]+marginal[g]*cap[g]/10.
            if not available[t,g]:linear[i]+=outage
            else:linear[i]+=lam*(cap[g]**2-2*demand[t]*cap[g])
        for pos,g in enumerate(active):
            for h in active[pos+1:]:_add_pair(pair,_q(g,t),_q(h,t),2*lam*cap[g]*cap[h])
    for g in range(NUM_GENERATORS):
        linear[_q(g,0)]+=startup[g]
        for t in range(1,NUM_PERIODS):
            linear[_q(g,t)]+=startup[g];_add_pair(pair,_q(g,t-1),_q(g,t),-startup[g])
    # x=(1-Z)/2 maps QUBO to H=sum h_i Z_i + sum J_ij Z_iZ_j + constant.
    h=-.5*linear.copy();j={}
    for (i,k),value in pair.items():h[i]-=value/4;h[k]-=value/4;j[(i,k)]=value/4
    scale=max(1.,float(np.max(np.abs(h))),max((abs(v) for v in j.values()),default=0.))
    return linear,pair,h/scale,{edge:value/scale for edge,value in j.items()},\
           {'demand_penalty':lam,'outage_penalty':outage,'hamiltonian_normalization':scale}

def _warm_start(instance,rng):
    cap=instance['capacity_mw'];cost=instance['marginal_cost'];available=instance['availability'];demand=instance['demand_mw']
    schedule=np.zeros((NUM_PERIODS,NUM_GENERATORS),dtype=int)
    for t in range(NUM_PERIODS):
        supplied=0.
        for g in sorted(range(NUM_GENERATORS),key=lambda x:cost[x]):
            if not available[t,g]:continue
            schedule[t,g]=1;supplied+=cap[g]
            if supplied>=demand[t]:break
    probabilities=np.where(schedule.reshape(-1)==1,.82,.18)
    probabilities=np.clip(probabilities+rng.uniform(-.04,.04,NUM_QUBITS),.08,.92)
    return schedule,probabilities,2*np.arcsin(np.sqrt(probabilities))

def build_uc_qaoa16(index,master_seed=DEFAULT_MASTER_SEED,p_range=DEFAULT_P_RANGE):
    low,high=map(int,p_range)
    if low<1 or high<low:raise ValueError('p_range must satisfy 1 <= low <= high')
    seed=stable_seed(f'uc_qaoa16_{master_seed}_{index}',29002);rng=np.random.default_rng(seed)
    p=int(rng.integers(low,high+1));instance=_uc_instance(rng)
    linear,pair,fields,couplings,penalties=_uc_qubo(instance,rng)
    schedule,probabilities,angles=_warm_start(instance,rng)
    gammas=rng.uniform(.08,1.10,p);betas=rng.uniform(.08,.85,p)
    qc=QuantumCircuit(NUM_QUBITS,name=f'uc_qaoa16_p{p}_{index}')
    for q,theta in enumerate(angles):qc.ry(float(theta),q)
    for layer in range(p):
        gamma=float(gammas[layer]);beta=float(betas[layer])
        for (a,b),coupling in sorted(couplings.items()):
            qc.cx(a,b);qc.rz(float(2*gamma*coupling),b);qc.cx(a,b)
        for q,field in enumerate(fields):qc.rz(float(2*gamma*field),q)
        for q in range(NUM_QUBITS):qc.rx(float(2*beta),q)
    info={'generator':'Qiskit QuantumCircuit Unit-Commitment QUBO-QAOA','problem':'unit commitment',
      'seed':int(seed),'master_seed':int(master_seed),'qaoa_layers_p':p,'p_range':[low,high],
      'num_generators':NUM_GENERATORS,'num_periods':NUM_PERIODS,'variable_mapping':'qubit=4*time_period+generator',
      'capacity_mw':[float(x) for x in instance['capacity_mw']],
      'marginal_cost':[float(x) for x in instance['marginal_cost']],
      'no_load_cost':[float(x) for x in instance['no_load_cost']],
      'startup_cost':[float(x) for x in instance['startup_cost']],
      'demand_mw':[float(x) for x in instance['demand_mw']],
      'availability':instance['availability'].astype(int).tolist(),
      'qubo_linear':[float(x) for x in linear],
      'qubo_quadratic':{f'{a}-{b}':float(v) for (a,b),v in sorted(pair.items())},
      'ising_z_fields_normalized':[float(x) for x in fields],
      'ising_zz_couplings_normalized':{f'{a}-{b}':float(v) for (a,b),v in sorted(couplings.items())},
      'qubo_interaction_count':len(couplings),'penalties':penalties,
      'warm_start_schedule':schedule.astype(int).tolist(),
      'warm_start_probabilities':[float(x) for x in probabilities],
      'warm_start_ry_angles':[float(x) for x in angles],
      'gammas':[float(x) for x in gammas],'betas':[float(x) for x in betas],
      'logical_depth':int(qc.depth()),'logical_gate_count':int(qc.size())}
    return qc,info

def make_sample(index,split,master_seed=DEFAULT_MASTER_SEED,p_range=DEFAULT_P_RANGE,shots=SHOTS):
    logical,info=build_uc_qaoa16(index,master_seed,p_range)
    noise=make_uc_physical_noise(f'uc_qaoa16_{master_seed}_{split}_{index}')
    distributed,events,operations=build_dynamic_tp(logical,noise);obs=observable_definitions()
    ideal=exact_logical_expectations(logical,obs)
    noiseless,_=run_dynamic_once(distributed,obs,shots,stable_seed(f'uc16_nl_{index}',29010),None)
    noisy,counts=run_dynamic_once(distributed,obs,shots,stable_seed(f'uc16_noisy_{index}',29011),noise_model(noise,operations))
    records=gate_records(logical,noise);twoq=sum(r['num_qubits']==2 for r in records);remote=len(events)
    distances=[e['distance_km'] for e in events];qasm=qasm2.dumps(logical)
    delta=max(abs(ideal[n]-noiseless[n]) for n in OBSERVABLE_NAMES);tol=max(.10,6/np.sqrt(shots))
    features={'num_data_qubits':16,'num_physical_qubits':N_PHYSICAL,'num_qpus':4,'qubits_per_qpu':4,
      'qaoa_layers_p':info['qaoa_layers_p'],'uc_qubo_interaction_count':info['qubo_interaction_count'],
      'logical_depth':int(logical.depth()),'logical_gate_count':int(logical.size()),
      'distributed_depth':int(distributed.depth()),'distributed_gate_count':int(distributed.size()),
      'one_qubit_gate_count':sum(r['num_qubits']==1 for r in records),'two_qubit_gate_count':twoq,
      'remote_gate_count':remote,'local_gate_count':len(records)-remote,
      'communication_ratio_all_gates':remote/max(1,len(records)),
      'communication_ratio_two_qubit':remote/max(1,twoq),
      'communication_distance_total_km':float(sum(distances)),
      'communication_distance_mean_km':float(np.mean(distances)) if distances else 0.,
      'tp_measurement_count':4*remote,'tp_reset_count':4*remote,
      'gate_type_counts':{str(k):int(v) for k,v in logical.count_ops().items()}}
    return {'schema_version':SCHEMA_VERSION,'sample_id':f'uc_qaoa16_{split}_{index:05d}','split':split,
      'benchmark':'qiskit-unit-commitment-qubo-qaoa-variable-depth','generator_info':info,
      'num_data_qubits':16,'num_physical_qubits':N_PHYSICAL,'num_qpus':4,'shots':int(shots),
      'observable_definitions':obs,'observable_names':list(OBSERVABLE_NAMES),'single_z_names':list(SINGLE_Z_NAMES),
      'selected_zz_edges':[list(e) for e in ZZ_EDGES],'zz_names':list(ZZ_NAMES),'global_z_name':GLOBAL_Z_NAME,
      'logical_ideal':ideal,'distributed_noiseless':noiseless,'distributed_noisy':noisy,'mitigated':dict(noisy),
      'noiseless_validation':{'max_abs_delta':float(delta),'tolerance':float(tol),'passed':bool(delta<=tol),'shot_based':True},
      'logical_circuit':logical,'distributed_circuit':distributed,'logical_qasm':qasm,
      'qasm_sha256':hashlib.sha256(qasm.encode()).hexdigest(),'qpu_map':dict(QPU_MAP),
      'ancilla_resources':{'global_reusable_pool':[16,17,18]},'auxiliary_qubits_reused':True,
      'tp_simulation':'dynamic measurement and classical feed-forward','simulation_device':'GPU','noise':noise,
      'gate_records':records,'communication_events':events,'distributed_operation_records':operations,
      'circuit_features':features,'noisy_counts':counts}

def preflight(master_seed=DEFAULT_MASTER_SEED,p_range=DEFAULT_P_RANGE,num_probes=16):
    hashes=set();depths=[];layers=[];interactions=[];rows=[];obs=observable_definitions()
    for i in range(int(num_probes)):
        qc,info=build_uc_qaoa16(i,master_seed,p_range);hashes.add(hashlib.sha256(qasm2.dumps(qc).encode()).hexdigest())
        depths.append(qc.depth());layers.append(info['qaoa_layers_p']);interactions.append(info['qubo_interaction_count'])
        values=exact_logical_expectations(qc,obs);rows.append([values[n] for n in OBSERVABLE_NAMES])
    a=np.asarray(rows)
    return {'unique_circuits':len(hashes),'num_probes':int(num_probes),'qaoa_layers_seen':sorted(set(layers)),
      'logical_depth_range':[int(min(depths)),int(max(depths))],
      'qubo_interaction_count_range':[int(min(interactions)),int(max(interactions))],
      'ideal_min':float(a.min()),'ideal_max':float(a.max()),'overall_std':float(a.std()),
      'single_z_overall_std':float(a[:,:16].std()),'zz_overall_std':float(a[:,16:24].std()),
      'global_z_std':float(a[:,-1].std())}

def generate_dataset(root,split_sizes=SPLITS,master_seed=DEFAULT_MASTER_SEED,p_range=DEFAULT_P_RANGE,shots=SHOTS,force=False):
    devices=require_gpu();root=Path(root);root.mkdir(parents=True,exist_ok=True);report=preflight(master_seed,p_range)
    meta={'schema_version':SCHEMA_VERSION,'problem':'unit commitment','generator':'Qiskit UC-QUBO-QAOA',
      'simulation_device':'GPU','aer_available_devices':list(devices),'num_data_qubits':16,'num_physical_qubits':N_PHYSICAL,
      'num_qpus':4,'qubits_per_qpu':4,'num_generators':4,'num_periods':4,'split_sizes':dict(split_sizes),
      'samples_per_pk':100,'master_seed':int(master_seed),'p_range':list(p_range),'shots':int(shots),
      'training_outputs':list(OBSERVABLE_NAMES),'preflight':report}
    (root/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');index=0
    for split,size in split_sizes.items():
        folder=root/split;folder.mkdir(parents=True,exist_ok=True)
        for start in range(0,int(size),100):
            path=folder/f'step_{start//100}.pk';batch=min(100,int(size)-start)
            if path.exists() and not force:print('skip',path);index+=batch;continue
            rows=[]
            for _ in tqdm(range(batch),desc=f'UC-QAOA16 dynamic TP GPU {split} {start}'):
                row=make_sample(index,split,master_seed,p_range,shots)
                if not row['noiseless_validation']['passed']:raise AssertionError(row['sample_id'],row['noiseless_validation'])
                rows.append(row);index+=1
            tmp=path.with_suffix('.pk.tmp')
            with tmp.open('wb') as f:pickle.dump({'schema_version':SCHEMA_VERSION,'split':split,'sample_count':len(rows),'samples':rows},f,pickle.HIGHEST_PROTOCOL)
            tmp.replace(path);print('saved',path)
    return meta
