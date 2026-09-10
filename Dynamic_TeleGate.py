"""19-qubit dynamic-teleportation random dataset, with mandatory Aer GPU."""
from __future__ import annotations
import json,pickle
from pathlib import Path
import numpy as np
from qiskit import ClassicalRegister,QuantumCircuit,QuantumRegister,transpile
from qiskit.quantum_info import Pauli,Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel,ReadoutError,depolarizing_error,thermal_relaxation_error,pauli_error
from tqdm.auto import tqdm
from DQC_Noise_Model import stable_seed
from DQC_Noise_Model import (NUM_QUBITS,NUM_QPUS,QPU_MAP,ZZ_EDGES,ZZ_NAMES,SPLITS,
 SAMPLES_PER_PK,SHOTS,DEFAULT_DEPTH_RANGE,DEFAULT_MASTER_SEED,build_random16,
 make_heterogeneous_noise,gate_records,observable_definitions)

N_AUX=3;N_PHYSICAL=NUM_QUBITS+N_AUX;A_BELL=16;A_REMOTE=17;A_RETURN=18
SCHEMA_VERSION='finally_random16_dynamic_measurement_tp_gpu_700_v5'

def exact_logical_expectations(circuit,observables):
    state=Statevector.from_instruction(circuit);values={}
    for name,active in observables.items():
        label=['I']*circuit.num_qubits
        for q in active:label[circuit.num_qubits-1-q]='Z'
        values[name]=float(np.real(state.expectation_value(Pauli(''.join(label)))))
    return values

def require_gpu():
    try:devices=tuple(AerSimulator().available_devices())
    except Exception as exc:raise RuntimeError('Cannot query Aer devices') from exc
    if not any(str(x).upper()=='GPU' for x in devices):
        raise RuntimeError(f'Aer GPU is required but available_devices={devices}. Install qiskit-aer-gpu/CUDA.')
    return devices

def _teleport(qc,source,bell,target,c0,c1,operations,pair_key,stage,gate_id):
    qc.reset(bell);qc.reset(target)
    operations.extend([{'name':'reset','qubits':[bell]},{'name':'reset','qubits':[target]}])
    qc.h(bell);qc.cx(bell,target);qc.cx(source,bell);qc.h(source)
    operations.extend([{'name':'h','qubits':[bell]},{'name':'cx','qubits':[bell,target]},
                       {'name':'cx','qubits':[source,bell]},{'name':'h','qubits':[source]}])
    qc.measure(source,c0);qc.measure(bell,c1)
    operations.extend([{'name':'measure','qubits':[source]},{'name':'measure','qubits':[bell]}])
    with qc.if_test((qc.clbits[c1],True)):qc.x(target)
    with qc.if_test((qc.clbits[c0],True)):qc.z(target)
    operations.extend([{'name':'x','qubits':[target]},{'name':'z','qubits':[target]}])
    for op in operations[-10:]:
        op.update({'kind':'tp_internal','noise_class':'communication','communication_pair_key':pair_key,
                   'tp_stage':stage,'logical_gate_id':gate_id})

def build_dynamic_tp(logical,noise):
    q=QuantumRegister(N_PHYSICAL,'q');c=ClassicalRegister(4,'teleport')
    out=QuantumCircuit(q,c,name=f'dynamic_tp_{logical.name}');events=[];operations=[]
    for gate_id,item in enumerate(logical.data):
        qs=[logical.find_bit(x).index for x in item.qubits]
        remote=len(qs)==2 and QPU_MAP[qs[0]]!=QPU_MAP[qs[1]]
        if not remote:
            out.append(item.operation,qs);operations.append({'kind':'local_gate','name':item.operation.name,
             'qubits':qs,'noise_class':'local','logical_gate_id':gate_id});continue
        q1,q2=qs;source,target=QPU_MAP[q1],QPU_MAP[q2];pair=tuple(sorted((source,target)))
        pair_key=f'{pair[0]}-{pair[1]}';comm=noise['communication_profiles'][pair_key]
        _teleport(out,q1,A_BELL,A_REMOTE,0,1,operations,pair_key,'outbound',gate_id)
        out.append(item.operation,[A_REMOTE,q2]);operations.append({'kind':'remote_logical_gate',
         'name':item.operation.name,'qubits':[A_REMOTE,q2],'logical_qubits':[q1,q2],
         'noise_class':'communication','communication_pair_key':pair_key,'logical_gate_id':gate_id})
        _teleport(out,A_REMOTE,A_RETURN,q1,2,3,operations,pair_key,'return',gate_id)
        t2=min(noise['physical_noise_profile'][q]['t2_ns'] for q in (q1,q2))
        events.append({'gate_id':gate_id,'gate_type':item.operation.name,'logical_qubits':[q1,q2],
         'source_qpu':source,'target_qpu':target,'qpu_pair':list(pair),'distance_km':comm['distance_km'],
         'communication_noise':comm['pcomm'],'communication_duration_ns':comm['duration_ns'],
         'idle_decoherence':float(1-np.exp(-comm['duration_ns']/t2)),
         'outbound_auxiliaries':[A_BELL,A_REMOTE],'return_auxiliary':A_RETURN,
         'ancillas_reused':True,'bell_measurements':4,'classical_corrections':4,
         'remote_gate_supported_generically':True,'tp_representation':'dynamic measurement + classical feed-forward'})
    return out,events,operations

def _extend_aux_profiles(noise):
    t1=float(np.mean([x['mean_t1_us'] for x in noise['qpu_profiles']]))
    t2=float(min(np.mean([x['mean_t2_us'] for x in noise['qpu_profiles']]),2*t1))
    for q,role in ((A_BELL,'bell'),(A_REMOTE,'remote_work'),(A_RETURN,'return_bell')):
        noise['physical_noise_profile'].append({'qubit':q,'role':role,'t1_us':t1,'t2_us':t2,
                                                't1_ns':t1*1000,'t2_ns':t2*1000})
    pcomm=[x['pcomm'] for x in noise['communication_profiles'].values()]
    durations=[x['duration_ns'] for x in noise['communication_profiles'].values()]
    noise['effective_gpu_simulation_pcomm']=float(np.mean(pcomm))
    noise['effective_gpu_simulation_communication_duration_ns']=float(np.mean(durations))
    noise['bell_measurement_error']=float(np.clip(.025+.15*np.mean(pcomm),.025,.055))
    noise['reset_error']=float(np.clip(.004+.04*np.mean(pcomm),.004,.012))
    return noise

def _relax(profile,q,d):
    r=profile[q];return thermal_relaxation_error(r['t1_ns'],min(r['t2_ns'],2*r['t1_ns']),d)

def noise_model(noise,operations):
    model=NoiseModel();profile=noise['physical_noise_profile'];grouped={}
    for op in operations:
        name=op['name'];qs=tuple(op['qubits'])
        if name in ('measure','reset') or len(qs) not in (1,2):continue
        grouped.setdefault((name,qs),[]).append(op['noise_class'])
    effective={}
    for (name,qs),kinds in grouped.items():
        probabilities=[];durations=[]
        for kind in kinds:
            if kind=='communication':
                probabilities.append(noise['effective_gpu_simulation_pcomm'])
                durations.append(noise['effective_gpu_simulation_communication_duration_ns'])
            else:
                rows=[noise['qpu_profiles'][QPU_MAP[q]] for q in qs]
                key='p2' if len(qs)==2 else 'p1'
                probabilities.append(float(np.mean([row[key] for row in rows])))
                durations.append(noise['local_2q_duration_ns' if len(qs)==2 else 'local_1q_duration_ns'])
        p=float(np.mean(probabilities));d=float(np.mean(durations))
        effective[f'{name}:{list(qs)}']={'occurrences':len(kinds),
         'local_occurrences':kinds.count('local'),'communication_occurrences':kinds.count('communication'),
         'effective_error_rate':p,'effective_duration_ns':d}
        relax=_relax(profile,qs[0],d)
        for q in qs[1:]:relax=relax.tensor(_relax(profile,q,d))
        model.add_quantum_error(depolarizing_error(p,len(qs)).compose(relax),name,list(qs))
    noise['effective_instruction_noise']=effective
    reset=pauli_error([('X',noise['reset_error']),('I',1-noise['reset_error'])])
    for q in range(N_PHYSICAL):model.add_quantum_error(reset,'reset',[q])
    for q in range(NUM_QUBITS):
        p=noise['qpu_profiles'][QPU_MAP[q]]['readout_error']
        model.add_readout_error(ReadoutError([[1-p,p],[p,1-p]]),[q])
    p=noise['bell_measurement_error']
    for q in (A_BELL,A_REMOTE,A_RETURN):model.add_readout_error(ReadoutError([[1-p,p],[p,1-p]]),[q])
    return model
def _from_counts(counts,observables):
    total=sum(counts.values());values={}
    for name,active in observables.items():
        value=0.
        for key,count in counts.items():
            data=key.split()[0];bits=[int(x) for x in data[::-1]]
            value+=(1. if sum(bits[q] for q in active)%2==0 else -1.)*count/total
        values[name]=float(value)
    return values

def run_dynamic_once(circuit,observables,shots,seed,model=None):
    require_gpu();run=circuit.copy();result=ClassicalRegister(NUM_QUBITS,'result')
    run.add_register(result);run.measure(range(NUM_QUBITS),result)
    sim=AerSimulator(method='statevector',device='GPU',noise_model=model,seed_simulator=seed)
    compiled=transpile(run,sim,optimization_level=0,seed_transpiler=seed)
    counts=sim.run(compiled,shots=shots).result().get_counts()
    return _from_counts(counts,observables),{str(k):int(v) for k,v in counts.items()}

def make_sample(index,split,master_seed=DEFAULT_MASTER_SEED,depth_range=DEFAULT_DEPTH_RANGE,shots=SHOTS):
    logical,generator=build_random16(index,master_seed,depth_range)
    noise=_extend_aux_profiles(make_heterogeneous_noise(f'{master_seed}_{split}_{index}'))
    distributed,events,operations=build_dynamic_tp(logical,noise);obs=observable_definitions()
    ideal=exact_logical_expectations(logical,obs)
    noiseless,_=run_dynamic_once(distributed,obs,shots,stable_seed(f'nl_{index}',25010),None)
    noisy,counts=run_dynamic_once(distributed,obs,shots,stable_seed(f'noisy_{index}',25011),noise_model(noise,operations))
    delta=max(abs(ideal[n]-noiseless[n]) for n in ZZ_NAMES);tol=max(.08,5/np.sqrt(shots))
    records=gate_records(logical,noise);twoq=sum(r['num_qubits']==2 for r in records)
    features={'num_data_qubits':NUM_QUBITS,'num_physical_qubits':N_PHYSICAL,'num_tp_ancillas':N_AUX,
     'num_qpus':NUM_QPUS,'logical_depth':int(logical.depth()),'logical_gate_count':int(logical.size()),
     'distributed_depth':int(distributed.depth()),'distributed_gate_count':int(distributed.size()),
     'two_qubit_gate_count':twoq,'remote_gate_count':len(events),'local_gate_count':len(records)-len(events),
     'communication_ratio_two_qubit':len(events)/max(1,twoq),'tp_measurement_count':4*len(events),
     'tp_reset_count':4*len(events),'gate_type_counts':{str(k):int(v) for k,v in logical.count_ops().items()}}
    return {'schema_version':SCHEMA_VERSION,'sample_id':f'random16_{split}_{index:05d}','split':split,
     'benchmark':'independent-seeded-random-circuit-dynamic-tp-gpu','generator_info':generator,
     'num_data_qubits':NUM_QUBITS,'num_physical_qubits':N_PHYSICAL,'num_qpus':NUM_QPUS,'shots':shots,
     'observable_definitions':obs,'selected_zz_edges':[list(e) for e in ZZ_EDGES],'zz_names':list(ZZ_NAMES),
     'logical_ideal':ideal,'distributed_noiseless':noiseless,'distributed_noisy':noisy,'mitigated':dict(noisy),
     'noiseless_validation':{'max_abs_delta':delta,'tolerance':tol,'passed':delta<=tol,'shot_based':True},
     'logical_circuit':logical,'distributed_circuit':distributed,'qpu_map':dict(QPU_MAP),
     'ancilla_resources':{'global_reusable_pool':[A_BELL,A_REMOTE,A_RETURN]},'auxiliary_qubits_reused':True,
     'tp_simulation':'dynamic measurement and classical feed-forward','simulation_device':'GPU',
     'noise':noise,'gate_records':records,'communication_events':events,
     'distributed_operation_records':operations,'circuit_features':features,'noisy_counts':counts}

def generate_dataset(root,split_sizes=SPLITS,master_seed=DEFAULT_MASTER_SEED,depth_range=DEFAULT_DEPTH_RANGE,shots=SHOTS,force=False):
    devices=require_gpu();root=Path(root);root.mkdir(parents=True,exist_ok=True)
    meta={'schema_version':SCHEMA_VERSION,'tp_simulation':'dynamic measurement and classical feed-forward',
     'simulation_device':'GPU','aer_available_devices':list(devices),'num_data_qubits':NUM_QUBITS,
     'num_physical_qubits':N_PHYSICAL,'num_qpus':NUM_QPUS,'split_sizes':dict(split_sizes),
     'samples_per_pk':SAMPLES_PER_PK,'master_seed':master_seed,'requested_depth_range':list(depth_range),
     'shots':shots,'training_outputs':list(ZZ_NAMES)}
    (root/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');index=0
    for split,size in split_sizes.items():
        folder=root/split;folder.mkdir(parents=True,exist_ok=True)
        for start in range(0,int(size),SAMPLES_PER_PK):
            path=folder/f'step_{start//SAMPLES_PER_PK}.pk'
            if path.exists() and not force:print('skip',path);index+=min(SAMPLES_PER_PK,size-start);continue
            rows=[]
            for _ in tqdm(range(start,min(start+SAMPLES_PER_PK,size)),desc=f'dynamic TP GPU {split} {start}'):
                row=make_sample(index,split,master_seed,depth_range,shots)
                if not row['noiseless_validation']['passed']:raise AssertionError(row['noiseless_validation'])
                rows.append(row);index+=1
            tmp=path.with_suffix('.pk.tmp')
            with tmp.open('wb') as f:pickle.dump({'schema_version':SCHEMA_VERSION,'split':split,'sample_count':len(rows),'samples':rows},f,pickle.HIGHEST_PROTOCOL)
            tmp.replace(path);print('saved',path)
    return meta
