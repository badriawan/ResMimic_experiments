"""Stage5: actual checkout MotionLib/MotionLibHOI CPU validation, no simulator.

Uses the real Isaac Gym import graph before Torch, but never constructs a gym,
environment, policy, optimizer, renderer or physical scene.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import ROOT,REPO,SOURCE,DATA,install,sha,write_json,motion_root
install()
sys.path[:0]=[str(REPO/'pose'),str(REPO/'legged_gym'),str(REPO/'rsl_rl')]
import isaacgym  # Actual SDK must precede Torch.
import torch
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from pose.utils.motion_lib_pkl import MotionLib
from pose.utils.motion_lib_hoi import MotionLibHOI
from lift_hold_reference_loader import install_reference_interpolation,restore_reference_interpolation,self_test
from lift_hold_validation_common import pose_samples,world_fk,rotation_delta,stats,linear,contact_assessment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--sample-rate',type=int,default=600)
    args=parser.parse_args()
    path=args.motion.resolve(strict=True);report_path=args.report.resolve()
    output_root=motion_root(path)
    if ROOT not in report_path.parents:
        raise ValueError('Motion/report paths violate conversion boundary')
    started=time.monotonic()
    stock=[REPO/'pose/pose/utils/motion_lib_pkl.py',REPO/'pose/pose/utils/motion_lib_hoi.py',
           REPO/'pose/pose/utils/torch_utils.py',REPO/'assets/g1/g1_custom_collision_29dof.urdf']
    before={str(p):sha(p) for p in stock+[path,path.with_suffix('.npz')]}
    quality_path=ROOT/'config/quality_gates.json';quality_hash=sha(quality_path)
    fixture=self_test();token=install_reference_interpolation()
    kwargs=dict(motion_decompose=False,motion_smooth=False,motion_height_adjust=False)
    human=MotionLib(str(path),'cpu',**kwargs)
    hoi=MotionLibHOI(str(path),str(path.with_suffix('.npz')),'cpu',**kwargs)
    # Actual checkout import above supplies its NumPy2 pickle compatibility shim.
    with path.open('rb') as stream: motion=pickle.load(stream)
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z: obj={k:z[k].copy() for k in z.files}
    n=len(motion['root_pos']);fps=float(motion['fps']);duration=(n-1)/fps
    bodies=list(motion['link_body_list'])
    urdf=ET.parse(stock[-1]).getroot()
    joints=[j for j in urdf.findall('joint') if j.get('type')!='fixed'];names=[j.get('name') for j in joints]
    targets=json.loads((ROOT/'audit/dependencies_and_target.json').read_text())['target_contract']
    gates=dict(actual_loader_one_motion=human.num_motions()==hoi.num_motions()==1,
        frame_count_matches=int(hoi._motion_num_frames[0])==n,
        fps_matches=abs(float(hoi._motion_fps[0])-fps)<1e-6,
        duration_matches=abs(float(hoi.get_motion_length(torch.tensor([0]))[0])-duration)<2e-6,
        body_order_exact=bodies==targets['body_order'] and list(hoi._body_link_list)==bodies,
        target_dof_order_29=names==targets['dof_order'] and len(names)==29,
        robot_loader_values_exact=np.array_equal(human._motion_dof_pos.numpy(),motion['dof_pos'].astype(np.float32)),
        hoi_robot_values_exact=np.array_equal(hoi._motion_dof_pos.numpy(),motion['dof_pos'].astype(np.float32)),
        object_values_match=bool(np.allclose(hoi._object_root_pos.numpy(),obj['trans'],atol=1e-6,rtol=0)
                                 and np.allclose(hoi._object_root_rot.numpy(),obj['rot'],atol=1e-6,rtol=0)),
        interpolation_fixture_16_cases=fixture['status']=='PASS' and len(fixture['cases'])==16)
    count=int(np.ceil(duration*args.sample_rate))
    times=np.linspace(0,duration,count+1)
    query=torch.tensor(times[:-1],dtype=torch.float32);ids=torch.zeros(len(query),dtype=torch.long)
    # Below endpoint only; append the exported endpoint directly.
    actual=hoi.calc_hoi_motion_frame(ids.clone(),query.clone())
    robot=human.calc_motion_frame(ids.clone(),query.clone())
    gates['all_actual_samples_finite']=all(bool(torch.isfinite(v).all()) for v in actual+robot)
    gates['human_hoi_robot_channels_agree']=all(bool(torch.allclose(a,b,atol=2e-6,rtol=0)) for a,b in zip(actual[:9],robot))
    sampled=dict(root_pos=np.vstack([actual[0].numpy(),motion['root_pos'][-1]]),
                 root_rot=np.vstack([actual[1].numpy(),motion['root_rot'][-1]]),
                 dof_pos=np.vstack([actual[4].numpy(),motion['dof_pos'][-1]]))
    mathematical=pose_samples(motion,times)
    root_position_error=np.linalg.norm(sampled['root_pos']-mathematical['root_pos'],axis=1)
    root_rotation_error=rotation_delta(mathematical['root_rot'],sampled['root_rot'])
    dof_error=abs(sampled['dof_pos']-mathematical['dof_pos'])
    expected_obj_pos=linear(obj['trans'],times[:-1],fps)
    expected_obj_rot=Slerp(np.arange(n)/fps,Rotation.from_quat(obj['rot']))(times[:-1]).as_quat()
    object_position_error=np.linalg.norm(actual[-2].numpy()-expected_obj_pos,axis=1)
    object_rotation_error=rotation_delta(expected_obj_rot,actual[-1].numpy())
    # The actual API locates frames using float32 phase against its float32
    # duration. Separate that clock quantization from quaternion interpolation
    # error; keep the nominal float64-grid discrepancies reported above.
    loader_duration=np.float32(float(hoi.get_motion_length(torch.tensor([0]))[0]))
    phase32=np.clip(query.numpy()/loader_duration,np.float32(0),np.float32(1))
    coordinates32=phase32*np.float32(n-1)
    quantized_times=np.r_[coordinates32.astype(np.float64)/fps,duration]
    quantized_motion={k:(v.astype(np.float32) if isinstance(v,np.ndarray) and v.dtype.kind=='f' else v)
                      for k,v in motion.items()}
    phase_reference=pose_samples(quantized_motion,quantized_times)
    phase_errors=dict(root_position_m=np.linalg.norm(sampled['root_pos']-phase_reference['root_pos'],axis=1),
                      root_rotation_rad=rotation_delta(phase_reference['root_rot'],sampled['root_rot']),
                      dof_rad=abs(sampled['dof_pos']-phase_reference['dof_pos']))
    object_phase_pos=linear(obj['trans'].astype(np.float32),quantized_times[:-1],fps)
    object_phase_rot=Slerp(np.arange(n)/fps,Rotation.from_quat(obj['rot'].astype(np.float32)))(quantized_times[:-1]).as_quat()
    phase_errors['object_position_m']=np.linalg.norm(actual[-2].numpy()-object_phase_pos,axis=1)
    phase_errors['object_rotation_rad']=rotation_delta(object_phase_rot,actual[-1].numpy())
    gates.update(root_position_float32_phase_2um=float(phase_errors['root_position_m'].max())<2e-6,
                 root_rotation_float32_phase_3urad=float(phase_errors['root_rotation_rad'].max())<3e-6,
                 joint_float32_phase_3urad=float(phase_errors['dof_rad'].max())<3e-6,
                 object_position_float32_phase_2um=float(phase_errors['object_position_m'].max())<2e-6,
                 object_rotation_float32_phase_3urad=float(phase_errors['object_rotation_rad'].max())<3e-6)
    lp,lr,rr,world=world_fk(motion,urdf,names,bodies)
    gates['stored_local_FK_10um']=float(abs(lp-motion['local_body_pos']).max())<1e-5
    contact_gates,contact,dense,traces=contact_assessment(sampled,urdf,names,bodies,times)
    # Interpolated stored body positions and FK of interpolated DOFs are distinct
    # checkout channels; report their discrepancy rather than silently replacing.
    dense_lp=dense[0]
    stored_dense=np.concatenate([actual[6].numpy(),motion['local_body_pos'][-1:]],axis=0)
    interpolation_vs_fk=stats(np.linalg.norm(stored_dense-dense_lp,axis=-1))
    endpoint=torch.tensor([float(hoi.get_motion_length(torch.tensor([0]))[0])],dtype=torch.float32)
    original_endpoint=float(endpoint[0]);wrapped=hoi.calc_hoi_motion_frame(torch.tensor([0]),endpoint)
    gates['raw_endpoint_wrap_understood']=bool(abs(float(endpoint[0]))<1e-6 and torch.allclose(
        wrapped[-2][0],hoi._object_root_pos[0]+hoi._motion_root_pos_delta[0],atol=2e-6,rtol=0))
    trace_path=output_root/'validation'/(path.parent.name+'_'+path.stem+'_'+sha(path)[:10]+'_actual_loader_samples.npz')
    trace_path.parent.mkdir(parents=True,exist_ok=True)
    # Full link poses enable independent collision refinement in Python3.11.
    link_rotations=np.stack([dense[2]@dense[1][b] for b in bodies],axis=1)
    np.savez_compressed(trace_path,**traces,root_pos=sampled['root_pos'],root_rot=sampled['root_rot'],
        dof_pos=sampled['dof_pos'],world_link_pos=dense[3],world_link_rot=link_rotations,
        body_names=np.asarray(bodies),motion_sha256=np.asarray(sha(path)),source_motion=np.asarray(str(path)),
        object_pos=np.vstack([actual[-2].numpy(),obj['trans'][-1]]),
        object_rot=np.vstack([actual[-1].numpy(),obj['rot'][-1]]))
    inventory=json.loads((ROOT/'audit/source_inventory.json').read_text())
    source_checks={r['path']:sha(SOURCE/r['path'])==r['sha256'] for r in inventory['records']}
    gates['all_original_source_inventory_preserved']=all(source_checks.values())
    gates['checkout_and_candidate_files_preserved']=all(sha(p)==h for p,h in before.items())
    gates['quality_configuration_preserved']=sha(quality_path)==quality_hash
    restore_reference_interpolation(token)
    report=dict(status='PASS_ACTUAL_LOADER' if all(gates.values()) else 'FAIL_ACTUAL_LOADER',
        quality_status='PASS_SAMPLED_FOOT_CONTACT' if all(contact_gates.values()) else 'FAIL_SAMPLED_FOOT_CONTACT',
        scope='Actual checkout CPU loader and independent URDF FK only; no physical simulation or policy.',
        motion=str(path),motion_sha256=sha(path),frames=n,fps=fps,duration_s=duration,
        loader_classes=[MotionLib.__module__+'.'+MotionLib.__name__,MotionLibHOI.__module__+'.'+MotionLibHOI.__name__],
        sample_count=len(times),sample_rate_hz=args.sample_rate,gates=gates,
        contact_gates=contact_gates,contact=contact,
        interpolation_mode='Actual checkout classes with converter-local normalized shortest-path SLERP bindings',
        mathematical_interpolation_error=dict(root_position_m=stats(root_position_error),root_rotation_rad=stats(root_rotation_error),
            dof_rad=stats(dof_error),object_position_m=stats(object_position_error),object_rotation_rad=stats(object_rotation_error)),
        float32_phase_reference_error={k:stats(v) for k,v in phase_errors.items()},
        float32_phase_reference_scope='Independent SciPy SLERP/linear reference at NumPy float32 API phase coordinates; nominal float64 time drift remains reported separately.',
        stored_body_linear_interpolation_vs_dof_FK_m=interpolation_vs_fk,
        adapter_fixtures=fixture,endpoint=dict(raw_api_query_s=original_endpoint,mutated_query_s=float(endpoint[0]),
            behavior='wraps object to initial plus human root loop displacement; exact final arrays checked separately'),
        trace_file=str(trace_path),quality_config=str(quality_path),quality_config_sha256=quality_hash,
        immutable_hashes=before,source_inventory_checks=source_checks,
        environment=dict(python=sys.executable,numpy=np.__version__,torch=torch.__version__),
        elapsed_s=time.monotonic()-started,policy_executed=False,training_executed=False,physics_executed=False,
        limitations=['Format/loader PASS is separate from kinematic/contact/self-collision acceptance.',
            'Stock loader without the local adapter is not certified by this result.',
            'Per-frame delta channels remain FPS-sensitive; no checkpoint/policy success is inferred.'])
    write_json(report_path,report)
    print(json.dumps(dict(status=report['status'],quality_status=report['quality_status'],report=str(report_path),
                         failed_loader_gates=[k for k,v in gates.items() if not v],failed_contact_gates=[k for k,v in contact_gates.items() if not v])))
    if not all(gates.values()): raise SystemExit(1)


if __name__=='__main__': main()
