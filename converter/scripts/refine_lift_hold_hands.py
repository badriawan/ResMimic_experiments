"""Small world-hand translations resolve overlap without altering legs/clock.

Pure reference IK and static geometry queries; no policy, grasp-force model,
rendering, simulation stepping or training. All candidate failures are retained.
"""
import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import install,ROOT,REPO,DATA,sha,write_json
install()
import numpy as np
from scipy.spatial.transform import Rotation
from correct_lift_hold import Chain,solve
from lift_hold_validation_common import world_fk,pose_samples,stats
from lift_hold_collision_oracle import StaticCollisionOracle

HANDS=['left_rubber_hand','right_rubber_hand']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DATA/'candidates/attempt02/lift_hold_001_g1.pkl')
    parser.add_argument('--attempt',default='hand_attempt01')
    parser.add_argument('--shift',type=float,default=.0098)
    parser.add_argument('--rotation-deg',type=float,default=0.)
    args=parser.parse_args()
    if not 0<=args.shift<=.0099:raise ValueError('Translation must remain under frozen10mm total hand gate')
    if not 0<=args.rotation_deg<=2.:raise ValueError('Only the screened small rotation range is supported')
    source=args.input.resolve(strict=True);out=DATA/'candidates'/args.attempt
    if DATA not in source.parents or out.exists():raise ValueError('Input/output boundary or existing attempt')
    out.mkdir(parents=True)
    baseline=DATA/'baseline/lift_hold_001_g1.pkl'
    m=pickle.load(source.open('rb'));base=pickle.load(baseline.open('rb'))
    if m['fps']!=base['fps'] or len(m['root_pos'])!=len(base['root_pos']):raise ValueError('This refinement requires matching native baseline clock')
    urdf=ET.parse(REPO/'assets/g1/g1_custom_collision_29dof.urdf').getroot()
    joints=[j for j in urdf.findall('joint') if j.get('type')!='fixed'];names=[j.get('name') for j in joints]
    lower=np.array([float(j.find('limit').get('lower')) for j in joints]);upper=np.array([float(j.find('limit').get('upper')) for j in joints]);vmax=np.array([float(j.find('limit').get('velocity')) for j in joints])
    bodies=m['link_body_list'];n=len(m['root_pos']);fps=float(m['fps'])
    _,br,base_rr,base_wp=world_fk(base,urdf,names,bodies)
    rr=Rotation.from_quat(m['root_rot']).as_matrix()
    witness_path=DATA/'validation/baseline_hand_collision_witnesses.npz'
    with np.load(witness_path,allow_pickle=False) as z:
        worst=int(np.argmin(z['gap_m']));axis=z['normal_left_to_right'][worst].copy()
        points=[z['point_left'][worst].copy(),z['point_right'][worst].copy()]
    axis/=np.linalg.norm(axis)
    local_axes=[]
    for side,hand in enumerate(HANDS):
        lever=points[side]-base_wp[worst,bodies.index(hand)]
        rotation_axis=np.cross(lever,axis)*(-1 if side==0 else 1)
        rotation_axis/=np.linalg.norm(rotation_axis)
        local_axes.append((base_rr[worst]@br[hand][worst]).T@rotation_axis)
    candidate={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in m.items()}
    trace={};start=time.monotonic()
    for side,hand in enumerate(HANDS):
        chain=Chain(urdf,names,hand);indices=np.array(chain.indices[-7:]);previous=m['dof_pos'][0,indices].copy()
        position_target=base_wp[:,bodies.index(hand)].copy()+(-1 if side==0 else 1)*args.shift*axis
        rotation_target=(base_rr@br[hand])@Rotation.from_rotvec(local_axes[side]*np.deg2rad(args.rotation_deg)).as_matrix();rows=[]
        for k in range(n):
            values,pe,re,calls=solve(chain,m['dof_pos'][k],previous,m['root_pos'][k],rr[k],position_target[k],rotation_target[k],
                                   lower,upper,indices,velocity=vmax if k else None,fps=fps,max_nfev=150)
            candidate['dof_pos'][k,indices]=values;previous=values;rows.append([pe,re,calls])
        trace[hand]=dict(target_position_error_m=stats(np.asarray(rows)[:,0]),target_rotation_error_rad=stats(np.asarray(rows)[:,1]))
        print('IK',hand,trace[hand],flush=True)
    lp,lr,_,_=world_fk(candidate,urdf,names,bodies);candidate['local_body_pos']=lp
    path=out/'lift_hold_001_g1.pkl'
    with path.open('xb') as stream:pickle.dump(candidate,stream,protocol=4)
    shutil.copyfile(source.with_suffix('.npz'),path.with_suffix('.npz'))
    # Dense comparison uses independently interpolated baseline and candidate.
    duration=(n-1)/fps;times=np.linspace(0,duration,int(np.ceil(duration*600))+1)
    cs=pose_samples(candidate,times);bs=pose_samples(base,times)
    _,clr,crr,cwp=world_fk(cs,urdf,names,bodies)
    _,blr,brr,bwp=world_fk(bs,urdf,names,bodies)
    metrics={};all_pos=[];all_rot=[]
    for hand in HANDS:
        j=bodies.index(hand)
        pos=np.linalg.norm(cwp[:,j]-bwp[:,j],axis=1)
        angle=Rotation.from_matrix(np.swapaxes(brr@blr[hand],1,2)@(crr@clr[hand])).magnitude()
        metrics[hand]=dict(origin_deviation_vs_baseline_m=stats(pos),orientation_deviation_vs_baseline_rad=stats(angle))
        all_pos.extend(pos);all_rot.extend(angle)
    oracle=StaticCollisionOracle(links=HANDS)
    gaps=[];unresolved=0
    for k,t in enumerate(times):
        oracle.set_link_poses(np.stack([cwp[k,bodies.index(h)] for h in HANDS]),np.stack([crr[k]@clr[h][k] for h in HANDS]))
        v=oracle.pair_distance(*HANDS)
        gaps.append(v['distance_m'])
        unresolved+=int(not(v['signed_distance_valid'] and v['witness_gap_consistent']))
    q=candidate['dof_pos'];speed=abs(np.diff(q,axis=0))*fps
    gates=dict(root_path_exact=np.array_equal(candidate['root_pos'],m['root_pos']),root_orientation_exact=np.array_equal(candidate['root_rot'],m['root_rot']),
        legs_and_waist_exact=np.array_equal(candidate['dof_pos'][:,:15],m['dof_pos'][:,:15]),
        object_bytes_exact=sha(path.with_suffix('.npz'))==sha(source.with_suffix('.npz')),
        position_limits=not bool(((q<lower-1e-6)|(q>upper+1e-6)).any()),speed_limits=not bool((speed>vmax+1e-6).any()),
        total_hand_origin_within_10mm=max(all_pos)<=.01,total_hand_rotation_within_5deg=max(all_rot)<=np.deg2rad(5),
        hand_hand_gap_at_least_minus_1mm=min(gaps)>=-.001,hand_hand_distance_resolved=unresolved==0)
    report=dict(status='PASS_LOCAL_GATES_PENDING_FULL_COLLISION' if all(gates.values()) else 'FAIL',gates={k:bool(v) for k,v in gates.items()},
        input=str(source),input_sha256=sha(source),baseline=str(baseline),baseline_sha256=sha(baseline),
        witnesses=str(witness_path),witnesses_sha256=sha(witness_path),script_sha256=sha(Path(__file__)),
        output=str(path),output_sha256=sha(path),quality_gates_sha256=sha(ROOT/'config/quality_gates.json'),
        axis_selection='Constant worst baseline native-hand overlap witness; no per-frame normal jitter',axis_left_to_right=axis.tolist(),
        baseline_worst_frame=worst,world_translation_each_hand_m=args.shift,target_traces=trace,
        local_rotation_axes=np.asarray(local_axes).tolist(),additional_rotation_deg=args.rotation_deg,
        dense_hand_deviation=metrics,dense_hand_hand_gap_m=stats(gaps),worst_gap_time_s=float(times[np.argmin(gaps)]),
        unresolved_hand_distance_samples=unresolved,dense_samples=len(times),sample_rate_hz=600,
        maximum_joint_speed_rad_s=float(speed.max()),position_violation_count=int(((q<lower-1e-6)|(q>upper+1e-6)).sum()),
        speed_violation_count=int((speed>vmax+1e-6).sum()),elapsed_s=time.monotonic()-start,
        full_other_body_collision_validated=False,physical_grasp_verified=False,policy_or_training_run=False)
    write_json(ROOT/'reports'/f'hand_refine_{args.attempt}.json',report)
    np.savez_compressed(out/'hand_refinement_traces.npz',times_s=times,hand_hand_gap_m=np.asarray(gaps),axis=axis)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
