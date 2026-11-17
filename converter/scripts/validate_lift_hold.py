"""Independent array, FK, contact and static self-collision validation.

No dynamics, rendering, policy or training. Contact is a reference-kinematic
test; collision distances are source-shape proxies, not PhysX cooking proof.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from converter_common import ROOT, REPO, SOURCE, DATA, install, sha, write_json, motion_root
install()
import numpy as np
from scipy.spatial.transform import Rotation
from lift_hold_validation_common import stats, pose_samples, world_fk, contact_assessment, linear, rotation_delta
from scipy.spatial.transform import Slerp

URDF = REPO / 'assets/g1/g1_custom_collision_29dof.urdf'


def independent_mujoco_fk(motion,urdf,names,bodies):
    """A separate FK engine, with no contacts, dynamics or integration.

    Dummy inertials only permit movable MJCF bodies; they are never used for
    forces/balance and are not an exported robot/physical asset.
    """
    import mujoco
    root=ET.Element('mujoco',model='independent_urdf_kinematic_tree')
    ET.SubElement(root,'compiler',angle='radian',fusestatic='false')
    world=ET.SubElement(root,'worldbody')
    base=ET.SubElement(world,'body',name='pelvis')
    ET.SubElement(base,'freejoint',name='reference_root')
    ET.SubElement(base,'inertial',pos='0 0 0',mass='1',diaginertia='.01 .01 .01')
    mapped={'pelvis':base};pending=list(urdf.findall('joint'))
    while pending:
        following=[]
        for joint in pending:
            parent=joint.find('parent').get('link');child=joint.find('child').get('link')
            if parent not in mapped:
                following.append(joint);continue
            org=joint.find('origin')
            xyz=np.zeros(3) if org is None else np.fromstring(org.get('xyz','0 0 0'),sep=' ')
            rpy=np.zeros(3) if org is None else np.fromstring(org.get('rpy','0 0 0'),sep=' ')
            quat=Rotation.from_euler('xyz',rpy).as_quat()[[3,0,1,2]]
            child_body=ET.SubElement(mapped[parent],'body',name=child,
                pos=' '.join(map(str,xyz)),quat=' '.join(map(str,quat)))
            ET.SubElement(child_body,'inertial',pos='0 0 0',mass='1',diaginertia='.01 .01 .01')
            if joint.get('type')!='fixed':
                ET.SubElement(child_body,'joint',name=joint.get('name'),type='hinge',
                    axis=joint.find('axis').get('xyz'),limited='false')
            mapped[child]=child_body
        if len(following)==len(pending):raise ValueError('Disconnected URDF tree')
        pending=following
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model)
    addresses=[int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in names]
    body_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in bodies]
    expected=world_fk(motion,urdf,names,bodies)
    maximum_position=maximum_rotation=0.
    for i in range(len(motion['root_pos'])):
        data.qpos[:3]=motion['root_pos'][i];data.qpos[3:7]=motion['root_rot'][i,[3,0,1,2]]
        data.qpos[addresses]=motion['dof_pos'][i]
        mujoco.mj_kinematics(model,data)
        maximum_position=max(maximum_position,float(np.linalg.norm(data.xpos[body_ids]-expected[3][i],axis=-1).max()))
        expected_rot=np.stack([expected[2][i]@expected[1][b][i] for b in bodies])
        actual_rot=data.xmat[body_ids].reshape(-1,3,3)
        maximum_rotation=max(maximum_rotation,float(Rotation.from_matrix(np.swapaxes(expected_rot,1,2)@actual_rot).magnitude().max()))
    return dict(frames=len(motion['root_pos']),max_world_position_error_m=maximum_position,
                max_world_rotation_error_rad=maximum_rotation,mujoco_version=mujoco.__version__,
                method='Independent MuJoCo URDF-derived tree; mj_kinematics only, no mj_step or dynamics',
                physics_executed=False)


def load_contract(path):
    path = Path(path).resolve(strict=True)
    motion_root(path)
    with path.open('rb') as stream:
        motion = pickle.load(stream)
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
        obj = {key:z[key].copy() for key in z.files}
    urdf = ET.parse(URDF).getroot()
    joints = [j for j in urdf.findall('joint') if j.get('type') != 'fixed']
    names = [j.get('name') for j in joints]
    target = json.loads((ROOT/'audit/dependencies_and_target.json').read_text())['target_contract']
    return path,motion,obj,urdf,joints,names,target['body_order']


def array_gates(motion,obj,urdf,joints,names,bodies):
    n = len(motion['root_pos']); fps = float(motion['fps'])
    shapes = dict(root_pos=(n,3),root_rot=(n,4),dof_pos=(n,29),local_body_pos=(n,38,3))
    gates = dict(frame_count_at_least_3=n>=3,positive_finite_fps=bool(np.isfinite(fps) and fps>0),
                 target_29_dof_38_links=len(names)==29 and len(bodies)==38,
                 expected_shapes=all(np.asarray(motion[k]).shape==s for k,s in shapes.items()),
                 body_names_exact=list(motion['link_body_list'])==bodies,
                 finite_robot_arrays=all(np.isfinite(motion[k]).all() for k in shapes),
                 object_shape_matches=obj['trans'].shape==(n,3) and obj['rot'].shape==(n,4),
                 finite_object_arrays=bool(np.isfinite(obj['trans']).all() and np.isfinite(obj['rot']).all()),
                 source_time_shape=obj['source_time_s'].shape==(n,),
                 source_time_strictly_increasing=bool((np.diff(obj['source_time_s'])>0).all()),
                 root_unit_quaternion=bool(np.max(abs(np.linalg.norm(motion['root_rot'],axis=1)-1))<1e-6),
                 object_unit_quaternion=bool(np.max(abs(np.linalg.norm(obj['rot'],axis=1)-1))<1e-6))
    lp,lr,rr,world = world_fk(motion,urdf,names,bodies)
    fk_error = float(abs(lp-motion['local_body_pos']).max())
    gates['stored_local_FK_10um'] = fk_error < 1e-5
    q = np.asarray(motion['dof_pos']); speed=abs(np.diff(q,axis=0))*fps
    lower=np.array([float(j.find('limit').get('lower')) for j in joints])
    upper=np.array([float(j.find('limit').get('upper')) for j in joints])
    limits=np.array([float(j.find('limit').get('velocity')) for j in joints])
    violations=(q<lower-1e-6)|(q>upper+1e-6)
    gates['URDF_position_limits']=not bool(violations.any())
    gates['URDF_speed_limits']=bool((speed<=limits+1e-6).all())
    rows=[dict(name=names[i],minimum_rad=float(q[:,i].min()),maximum_rad=float(q[:,i].max()),
               lower_rad=float(lower[i]),upper_rad=float(upper[i]),speed_limit_rad_s=float(limits[i]),
               max_speed_rad_s=float(speed[:,i].max()),position_violations=int(violations[:,i].sum()),
               speed_violations=int((speed[:,i]>limits[i]+1e-6).sum())) for i in range(29)]
    diagnostics = dict(stored_local_FK_max_error_m=fk_error,joint_limits=rows,
        root_xy_displacement_max_m=float(np.linalg.norm(motion['root_pos'][:,:2]-motion['root_pos'][0,:2],axis=1).max()),
        root_height_m=stats(motion['root_pos'][:,2]),world_link_positions={})
    for name in ['left_rubber_hand','right_rubber_hand','left_elbow_link','right_elbow_link','left_ankle_roll_link','right_ankle_roll_link']:
        p=world[:,bodies.index(name)]
        diagnostics['world_link_positions'][name]=dict(min_xyz_m=p.min(axis=0).tolist(),max_xyz_m=p.max(axis=0).tolist(),
            max_displacement_from_first_m=float(np.linalg.norm(p-p[0],axis=1).max()))
    return gates,diagnostics


def collision_pairs():
    arms={s:[s+'_'+x for x in ['rubber_hand','wrist_yaw_link','elbow_link']] for s in ['left','right']}
    legs={s:[s+'_'+x for x in ['hip_yaw_link','knee_link','ankle_roll_link']] for s in ['left','right']}
    body=['torso_link']+legs['left']+legs['right']
    pairs=[(a,b) for s in arms for a in arms[s] for b in body]
    pairs += [(a,b) for a in arms['left'] for b in arms['right']]
    pairs += [(a,b) for a in legs['left'] for b in legs['right']]
    return pairs


def baseline_comparison(path,motion,obj,urdf,names,bodies,times,dense_fk):
    baseline_path=DATA/'baseline/lift_hold_001_g1.pkl'
    if path==baseline_path or not baseline_path.is_file():
        return {},dict(scope='Baseline itself; no additional correction')
    with baseline_path.open('rb') as stream: baseline=pickle.load(stream)
    with np.load(baseline_path.with_suffix('.npz'),allow_pickle=False) as z: old_obj={k:z[k].copy() for k in z.files}
    old_duration=(len(baseline['root_pos'])-1)/float(baseline['fps'])
    if abs(old_duration-times[-1])>1e-8:
        return dict(baseline_duration_preserved=False),dict(old_duration_s=old_duration,new_duration_s=float(times[-1]))
    old_sampled=pose_samples(baseline,times)
    old_fk=world_fk(old_sampled,urdf,names,bodies)
    sampled=pose_samples(motion,times)
    hand={}
    for name in ['left_rubber_hand','right_rubber_hand']:
        old_rotation=old_fk[2]@old_fk[1][name]
        new_rotation=dense_fk[2]@dense_fk[1][name]
        angles=np.degrees(Rotation.from_matrix(np.swapaxes(old_rotation,1,2)@new_rotation).magnitude())
        displacement=np.linalg.norm(dense_fk[3][:,bodies.index(name)]-old_fk[3][:,bodies.index(name)],axis=1)
        hand[name]=dict(origin_displacement_m=stats(displacement),rotation_difference_deg=stats(angles))
    old_positions=linear(old_obj['trans'],times,baseline['fps'])
    new_positions=linear(obj['trans'],times,motion['fps'])
    old_rotation=Slerp(np.arange(len(old_obj['rot']))/baseline['fps'],Rotation.from_quat(old_obj['rot']))(times).as_quat()
    new_rotation=Slerp(np.arange(len(obj['rot']))/motion['fps'],Rotation.from_quat(obj['rot']))(times).as_quat()
    position_error=np.linalg.norm(old_positions-new_positions,axis=1)
    rotation_error=rotation_delta(old_rotation,new_rotation)
    root_xy_error=np.linalg.norm(old_sampled['root_pos'][:,:2]-sampled['root_pos'][:,:2],axis=1)
    root_angle=rotation_delta(old_sampled['root_rot'],sampled['root_rot'])
    gates=dict(baseline_duration_preserved=True,
               baseline_root_xy_preserved=bool(root_xy_error.max()<1e-9),
               baseline_root_orientation_preserved=bool(root_angle.max()<1e-9),
               additional_hand_origin_correction_1cm=all(v['origin_displacement_m']['max']<=.01+1e-9 for v in hand.values()),
               additional_hand_rotation_correction_5deg=all(v['rotation_difference_deg']['max']<=5.+1e-7 for v in hand.values()),
               object_continuous_path_preserved=bool(position_error.max()<1e-8 and rotation_error.max()<1e-8),
               object_source_clock_endpoints_preserved=bool(np.allclose(obj['source_time_s'][[0,-1]],old_obj['source_time_s'][[0,-1]],atol=1e-10,rtol=0)))
    return gates,dict(baseline=str(baseline_path),hand=hand,root_xy_max_error_m=float(root_xy_error.max()),
                      root_rotation_max_error_rad=float(root_angle.max()),object_position_error_m=stats(position_error),
                      object_rotation_error_rad=stats(rotation_error),
                      root_z_change_m=stats(sampled['root_pos'][:,2]-old_sampled['root_pos'][:,2]))


def collision_assessment(motion,urdf,names,bodies,rate=30.,loader_samples=None,motion_hash=None):
    from lift_hold_collision_oracle import StaticCollisionOracle
    duration=(len(motion['root_pos'])-1)/float(motion['fps'])
    actual_world_rot=None
    if loader_samples is not None:
        with np.load(loader_samples,allow_pickle=False) as z:
            if str(z['motion_sha256'])!=motion_hash:
                raise ValueError('Loader samples do not match the candidate motion SHA256')
            if z['body_names'].tolist()!=bodies:
                raise ValueError('Loader sample body order differs from candidate')
            times=z['times_s'].copy();world=z['world_link_pos'].copy();actual_world_rot=z['world_link_rot'].copy()
        if not (np.isfinite(world).all() and np.isfinite(actual_world_rot).all() and (np.diff(times)>0).all()):
            raise ValueError('Invalid actual loader pose samples')
        rate=float(1/np.median(np.diff(times)))
    else:
        times=np.linspace(0,duration,int(np.ceil(duration*rate))+1)
        sampled=pose_samples(motion,times)
        lp,lr,rr,world=world_fk(sampled,urdf,names,bodies)
    pairs=collision_pairs(); links=sorted({x for pair in pairs for x in pair})
    oracle=StaticCollisionOracle(URDF,links=links)
    registration=all(v['passed'] for v in oracle.mesh_registration.values())
    results={a+'__'+b:dict(pair=[a,b],minimum_gap_m=1e6,negative_samples=0,unresolved_samples=0,
                         broad_sphere_proofs=0,narrow_queries=0,worst_time_s=None,worst_method=None)
             for a,b in pairs}
    negative_times=[]; tolerance=.001
    for index,t in enumerate(times):
        positions=np.stack([world[index,bodies.index(link)] for link in links])
        rotations=np.stack([actual_world_rot[index,bodies.index(link)] if actual_world_rot is not None
                            else rr[index]@lr[link][index] for link in links])
        oracle.set_link_poses(positions,rotations)
        for a,b in pairs:
            row=results[a+'__'+b]
            # A positive enclosing-sphere lower bound proves all shape pairs
            # separated. Close cases use the reviewed exact-support/native oracle.
            bounds=[]
            for ga in oracle.geom_ids[a]['capsule']:
                for gb in oracle.geom_ids[b]['capsule']:
                    bounds.append(float(np.linalg.norm(oracle.data.geom_xpos[ga]-oracle.data.geom_xpos[gb])
                                        -oracle.model.geom_rbound[ga]-oracle.model.geom_rbound[gb]))
            bound=min(bounds)
            if bound>=.005:
                gap=bound;method='enclosing_sphere_separation_lower_bound';valid=True
                row['broad_sphere_proofs']+=1
            else:
                variants=[oracle.pair_distance(a,b,v) for v in ['native','capsule']]
                selected=min(variants,key=lambda x:x['distance_m'])
                gap=float(selected['distance_m']);method=selected['distance_method']
                valid=all(v['signed_distance_valid'] and v['witness_gap_consistent'] for v in variants)
                row['narrow_queries']+=2
            if gap<row['minimum_gap_m']:
                row.update(minimum_gap_m=gap,worst_time_s=float(t),worst_method=method)
            if not valid:
                row['unresolved_samples']+=1
            if valid and gap < -tolerance:
                row['negative_samples']+=1
                negative_times.append(dict(time_s=float(t),pair=[a,b],gap_m=gap))
        if index%100==0 or index==len(times)-1:
            print('Static collision sample {}/{}'.format(index+1,len(times)),flush=True)
    gates=dict(source_mesh_registration=registration,
               no_sampled_source_shape_overlap=not bool(negative_times),
               no_unresolved_collision_queries=not any(v['unresolved_samples'] for v in results.values()))
    return gates,dict(samples=len(times),sample_rate_hz=rate,pair_count=len(pairs),tolerance_m=tolerance,
        pose_source='actual checkout loader DOF/root samples followed by independent FK' if loader_samples else 'mathematical linear/SLERP sampling followed by independent FK',
        loader_samples=str(loader_samples) if loader_samples else None,
        pairs=list(results.values()),negative_observations=negative_times,
        mesh_registration=oracle.mesh_registration,source_hashes=oracle.source_hashes,
        scope='Sampled nonadjacent distal-arm/torso/leg/cross-arm and cross-leg source convex-hull/primitive proxies.',
        limitations=['Finite-time source-shape collision sampling is not a continuous-time proof.',
                     'Positive bounding/support distances are lower bounds, not closest-point distances.',
                     'Not PhysX collision cooking, contact forces, balance or physical grasp validation.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--sample-rate',type=int,default=600)
    parser.add_argument('--collision-rate',type=float,default=30.)
    parser.add_argument('--skip-collisions',action='store_true')
    parser.add_argument('--loader-samples',type=Path,help='Use exact exported actual-loader poses for collision checks')
    args=parser.parse_args()
    report_path=args.report.resolve()
    if ROOT not in report_path.parents: raise ValueError('Report must be inside converter')
    started=time.monotonic()
    quality_path=ROOT/'config/quality_gates.json'
    quality=json.loads(quality_path.read_text())
    expected=dict(minimum_foot_native_and_capsule_gap_m=-.002,support_gap_abs_band_m=.005,
                  near_floor_probe_xy_speed_max_m_s=.1,near_floor_probe_xy_speed_p95_m_s=.05,
                  critical_nonadjacent_robot_shape_minimum_gap_m=-.001)
    if any(quality.get(k)!=v for k,v in expected.items()):
        raise ValueError('Quality thresholds differ from this independent validator implementation')
    path,motion,obj,urdf,joints,names,bodies=load_contract(args.motion)
    hashes={str(p):sha(p) for p in [path,path.with_suffix('.npz'),URDF]}
    array,diagnostics=array_gates(motion,obj,urdf,joints,names,bodies)
    independent_fk=independent_mujoco_fk(motion,urdf,names,bodies)
    array['independent_mujoco_FK_10um']=independent_fk['max_world_position_error_m']<1e-5
    array['independent_mujoco_FK_rotation_1urad']=independent_fk['max_world_rotation_error_rad']<1e-6
    diagnostics['independent_mujoco_FK']=independent_fk
    duration=(len(motion['root_pos'])-1)/float(motion['fps'])
    if args.loader_samples:
        loaded_path=args.loader_samples.resolve(strict=True)
        if DATA not in loaded_path.parents:raise ValueError('Actual loader samples must be under data_after_conversion')
        with np.load(loaded_path,allow_pickle=False) as z:
            if str(z['motion_sha256'])!=sha(path):raise ValueError('Actual loader sample motion hash mismatch')
            times=z['times_s'].copy()
            sampled={name:z[name].copy() for name in ['root_pos','root_rot','dof_pos']}
    else:
        times=np.linspace(0,duration,int(np.ceil(duration*args.sample_rate))+1)
        sampled=pose_samples(motion,times)
    q=sampled['dof_pos'];lower=np.array([float(j.find('limit').get('lower')) for j in joints])
    upper=np.array([float(j.find('limit').get('upper')) for j in joints]);speed_limit=np.array([float(j.find('limit').get('velocity')) for j in joints])
    sampled_speed=abs(np.diff(q,axis=0))/np.diff(times)[:,None]
    array['dense_URDF_position_limits']=bool(((q>=lower-1e-6)&(q<=upper+1e-6)).all())
    array['dense_URDF_speed_limits']=bool((sampled_speed<=speed_limit+1e-6).all())
    diagnostics['dense_joint_speed_max_rad_s']=sampled_speed.max(axis=0).tolist()
    contact,contact_report,dense_fk,traces=contact_assessment(sampled,urdf,names,bodies,times)
    correction_gates,correction_report=baseline_comparison(path,motion,obj,urdf,names,bodies,times,dense_fk)
    trace_path=DATA/'validation'/(path.parent.name+'_'+path.stem+'_'+sha(path)[:10]+'_geometry_samples.npz')
    trace_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(trace_path,**traces)
    report=dict(status='VALIDATING_COLLISIONS' if not args.skip_collisions else 'ARRAY_AND_CONTACT_ONLY',
        motion=str(path),motion_sha256=hashes[str(path)],frames=len(motion['root_pos']),fps=float(motion['fps']),
        duration_s=duration,array_gates=array,contact_gates=contact,diagnostics=diagnostics,
        quality_config=str(quality_path),quality_config_sha256=sha(quality_path),
        contact=contact_report,trace_file=str(trace_path),collision_gates=None,collision=None,
        correction_gates=correction_gates,baseline_comparison=correction_report,
        declared_thresholds=dict(foot_penetration_tolerance_m=.002,support_band_m=.005,
            near_floor_probe_speed_max_m_s=.1,near_floor_probe_speed_p95_m_s=.05,
            self_collision_overlap_tolerance_m=.001),
        policy_executed=False,training_executed=False,physics_executed=False)
    validation_sources=[Path(__file__),Path(__file__).with_name('lift_hold_validation_common.py'),
                        Path(__file__).with_name('lift_hold_collision_oracle.py'),
                        Path(__file__).with_name('test_collision_aggregation.py')]
    report['validation_source_sha256']={str(p):sha(p) for p in validation_sources if p.is_file()}
    write_json(report_path,report)
    if not args.skip_collisions:
        loader_samples=args.loader_samples.resolve(strict=True) if args.loader_samples else None
        if loader_samples is not None and DATA not in loader_samples.parents:
            raise ValueError('Loader sample data must stay under data_after_conversion')
        cgate,creport=collision_assessment(motion,urdf,names,bodies,args.collision_rate,loader_samples,sha(path))
        report.update(collision_gates=cgate,collision=creport)
    report['input_hashes_preserved']={str(p):sha(p)==value for p,value in hashes.items()}
    gates={**array,**contact,**correction_gates,**(report['collision_gates'] or {})}
    passed=all(gates.values()) and all(report['input_hashes_preserved'].values())
    report['status']=('PASS_ARRAY_CONTACT_ONLY' if args.skip_collisions else 'PASS_REFERENCE_KINEMATIC') if passed else 'FAIL_REFERENCE_KINEMATIC'
    report['collision_coverage']='NOT_TESTED' if args.skip_collisions else 'ACTUAL_LOADER_ALL_EXPORTED_SAMPLES' if args.loader_samples else 'SAMPLED_{}_HZ'.format(args.collision_rate)
    report['elapsed_s']=time.monotonic()-started
    write_json(report_path,report)
    print(json.dumps(dict(status=report['status'],report=str(report_path),failed_gates=[k for k,v in gates.items() if not v],
                         minimum_foot_gap_m=contact_report['minimum_gap_m'],unsupported_samples=contact_report['unsupported_samples'])))
    if not passed: raise SystemExit(1)


if __name__=='__main__': main()
