"""Stages 1--3 for the declared lift_hold_001 source; no policy or dynamics.

CPU alignment, source/mesh contract checks, actual GMR/Mink IK, and target URDF
FK. New data go only to data_after_conversion; configuration/reports stay here.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from converter_common import install, ROOT, REPO, SOURCE, DATA, sha, write_json
install()

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import trimesh

sys.path.insert(0, str(REPO / 'legged_gym/legged_gym/scripts'))
from convert_gmr_qpos import forward_kinematics, foot_clearance, clearance_summary

EPISODE = SOURCE / 'output/frl_apartment_4_sub17_floorlamp_001_0'
PRIMARY = EPISODE / 'sub17_floorlamp_001_pidx_0_task_0.npy'
RECORDER = EPISODE / 'maskedmimic_res/task_0/stage10_raw_trajectory.npz'
REST_MESH = Path('/home/badriawan/InterPose/InterPose/processed_data/rest_object_geo/floorlamp.ply')
URDF = REPO / 'assets/g1/g1_custom_collision_29dof.urdf'
XML = REPO / 'GMR/assets/unitree_g1/g1_mocap_29dof.xml'
ALIGNED = DATA / 'aligned/lift_hold_001_aligned.npz'
BASELINE = DATA / 'baseline/lift_hold_001_g1.pkl'
CONFIG = ROOT / 'config/lift_hold_001.json'


def stats(values):
    v = np.asarray(values, dtype=float)
    return dict(min=float(v.min()), median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max()))


def linear(values, source_time, query):
    shape = values.shape[1:]
    flat = values.reshape(len(source_time), -1)
    return np.stack([np.interp(query, source_time, flat[:,j]) for j in range(flat.shape[1])], axis=-1).reshape((len(query),)+shape)


def data_save(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def source_contract():
    d = np.load(PRIMARY, allow_pickle=True).item()  # Trusted user's generated project dictionary.
    metadata = d['metadata']
    human = d['human']['joint_xyz'][0]
    obj = d['object']['floorlamp']['xyz'][0]
    quat = d['object']['floorlamp']['rotation'][0]
    fps = float(metadata['fps'])
    origin = np.r_[human[0,0,:2],0.]
    if metadata['position_units'] != 'meter' or metadata['axis_angle_units'] != 'radian' or metadata['quaternion_order'].lower() != 'wxyz':
        raise ValueError('Source units/quaternion contract changed')
    rest = trimesh.load_mesh(REST_MESH, process=False)
    probes=[]
    for i in sorted(set([0,len(obj)//2,len(obj)-1])):
        mesh_path = EPISODE/'objs_mesh'/f'{i:05d}_object_floorlamp.ply'
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if not np.array_equal(rest.faces, mesh.faces) or len(rest.vertices) != len(mesh.vertices):
            raise ValueError('Rest and source object topology differ')
        expected = np.asarray(rest.vertices) @ Rotation.from_quat(quat[i,[1,2,3,0]]).as_matrix().T + obj[i]
        difference = np.asarray(mesh.vertices)-expected
        offset = difference.mean(axis=0)
        residual = difference-offset
        probes.append(dict(frame=i,mesh=str(mesh_path),mesh_sha256=sha(mesh_path),
            direct_max_vertex_error_m=float(np.linalg.norm(difference,axis=1).max()),
            fitted_constant_translation_m=offset.tolist(),
            max_vertex_residual_after_translation_m=float(np.linalg.norm(residual,axis=1).max())))
    offsets=np.asarray([p['fitted_constant_translation_m'] for p in probes])
    geometry_matches=max(p['max_vertex_residual_after_translation_m'] for p in probes)<1e-6
    constant_frame_offset=bool(np.max(abs(offsets-offsets[0]))<1e-6)
    if not geometry_matches or not constant_frame_offset:
        raise ValueError('Source object meshes cannot be reconciled as the same rigid geometry')
    # Primary NPY poses are the authoritative scene-normalized output. The PLY
    # visualization has a separately documented constant height offset. Never
    # apply that offset to the canonical poses or recenter the rest geometry.
    cfg=dict(dataset='lift_hold_001',source_root=str(SOURCE),primary_motion=str(PRIMARY),
        recorder=str(RECORDER),source_sha256=sha(PRIMARY),recorder_sha256=sha(RECORDER),
        output_data_root=str(DATA),report_root=str(ROOT/'reports'),scene_uniform_scale=1.,
        origin_xyz_m=origin.tolist(),transform='p_out = p_source - origin_xy; no scale or Z adjustment',
        known_source_initial_object_xyz_m=obj[0].tolist(),
        known_source_initial_object_xyz_transformed_m=(obj[0]-origin).tolist(),
        initial_phase_policy='Supported overlap only; t0 object initial excluded, no synthetic pose or timing shift',
        human_propotion_scaling='All GMR human_scale_table entries set to1.0',
        root_xy_policy='faithful source trajectory; do not freeze drift',
        alignment_fps=fps,clock_policy='measured human reference clock; derived object index/fps; supported overlap',
        source_to_target_quaternion='WXYZ -> XYZW',rest_object_mesh=str(REST_MESH),
        rest_object_mesh_sha256=sha(REST_MESH),target_urdf=str(URDF),target_urdf_sha256=sha(URDF),
        robot_xml=str(XML),robot_xml_sha256=sha(XML),
        boundary_roundoff_tolerance_s=1e-5,physical_success_verified=False,policy_or_training_run=False)
    write_json(CONFIG,cfg)
    # Identity transform of canonical geometry, exported as a portable reference.
    mesh_output=DATA/'assets/floorlamp.obj'
    mesh_output.parent.mkdir(parents=True,exist_ok=True)
    if mesh_output.exists():
        raise FileExistsError(mesh_output)
    rest.export(mesh_output)
    report=dict(status='PASS_WITH_DOCUMENTED_PLY_FRAME_OFFSET',config=str(CONFIG),config_sha256=sha(CONFIG),
        source_frames=len(human),fps=fps,source_position_shape=list(d['human']['joint_xyz'].shape),
        source_pose_shape=list(d['human']['pose_aa'].shape),primary_metadata=metadata,
        mesh_checks=probes,source_ply_constant_world_offset_m=offsets[0].tolist(),
        source_ply_frame_offset_applied_to_export=False,rest_vertices_recentered=False,
        geometry_same_after_constant_translation=True,
        mesh_interpretation='Reference geometry matches vertex identities/topology; source PLYs have a constant Z placement offset. Canonical primary pose + original rest geometry is retained.',
        scale=1.,source_lift_m=float(obj[:,2].max()-obj[0,2]),export_lift_m=float(obj[:,2].max()-obj[0,2]),
        root_xy_drift_preserved=True,physical_grasp_or_phase_sync_verified=False,
        mesh_output=str(mesh_output),mesh_output_sha256=sha(mesh_output))
    write_json(ROOT/'reports/stage1_contract.json',report)
    return d,cfg


def align(d,cfg):
    human=d['human']['joint_xyz'][0]
    aa=d['human']['pose_aa'][0]
    obj=d['object']['floorlamp']['xyz'][0]
    quat=d['object']['floorlamp']['rotation'][0]
    fps=cfg['alignment_fps'];tol=cfg['boundary_roundoff_tolerance_s']
    with np.load(RECORDER,allow_pickle=False) as z:
        ht=z['recorder_reference_time_s'].copy();st=z['recorder_sdk_time_s'].copy()
    ot=np.arange(len(obj),dtype=float)/fps
    assert len(ht)==len(human)==len(aa)==len(obj)
    assert (np.diff(ht)>0).all() and (np.diff(st)>0).all()
    low=max(ht[0],ot[0]);high=min(ht[-1],ot[-1])
    first=int(np.ceil((low-tol)*fps));last=int(np.floor((high+tol)*fps))
    times=np.arange(first,last+1,dtype=float)/fps
    qh=np.clip(times,ht[0],ht[-1]);qo=np.clip(times,ot[0],ot[-1])
    assert abs(qh-times).max()<tol and abs(qo-times).max()<tol
    hp=linear(human,ht,qh)
    haa=np.stack([Slerp(ht,Rotation.from_rotvec(aa[:,j]))(qh).as_rotvec() for j in range(aa.shape[1])],axis=1)
    op=linear(obj,ot,qo)
    oq=Slerp(ot,Rotation.from_quat(quat[:,[1,2,3,0]]))(qo).as_quat()
    oq=np.where(np.r_[False,np.sum(oq[1:]*oq[:-1],axis=1)<0][:,None],-oq,oq)
    parents=np.asarray(d['metadata']['pose_joint_parent_indices'],int)
    data_save(ALIGNED,human_joint_pos=hp,human_local_axis_angle=haa,object_pos=op,
        object_quat_wxyz=oq[:,[3,0,1,2]],fps=np.asarray(fps),source_time_s=times,
        human_query_time_s=qh,object_query_time_s=qo,human_recorder_time_s=ht,
        human_sdk_time_s=st,object_derived_time_s=ot,joint_parents22=parents,
        joint_names22=np.asarray(d['metadata']['pose_joint_names']))
    maxz=op[:,2].max();plateau=np.flatnonzero(np.isclose(op[:,2],maxz,atol=1e-6,rtol=0))
    withdrawal=d['metadata'].get('planned_hand_release_schedule',{})
    # Compare actual source tail movement and the canonical final pose, without
    # assuming planned annotations imply completed hand release.
    hand_end_motion={}
    for name,j in [('left_wrist',20),('right_wrist',21),('left_middle1',22),('right_middle1',23)]:
        hand_end_motion[name]=dict(source_last_interval_speed_m_s=float(np.linalg.norm(human[-1,j]-human[-2,j])/(ht[-1]-ht[-2])),
            aligned_last_interval_speed_m_s=float(np.linalg.norm(hp[-1,j]-hp[-2,j])*fps),
            dropped_tail_displacement_m=float(np.linalg.norm(human[-1,j]-hp[-1,j])))
    report=dict(status='PASS_NUMERIC_ALIGNMENT_WITH_END_PHASE_LIMITATION',source_frames=len(human),output_frames=len(times),
        fps=fps,source_human_time_first_last_s=ht[[0,-1]].tolist(),source_sdk_time_first_last_s=st[[0,-1]].tolist(),
        source_object_time_first_last_s=ot[[0,-1]].tolist(),source_output_time_first_last_s=times[[0,-1]].tolist(),
        endpoint_duration_s=float((len(times)-1)/fps),video_frame_duration_s=float(len(times)/fps),
        max_human_boundary_roundoff_clamp_s=float(abs(qh-times).max()),
        max_object_boundary_roundoff_clamp_s=float(abs(qo-times).max()),extrapolation_used=False,
        human_tail_time_not_exported_s=float(ht[-1]-times[-1]),object_initial_time_not_exported_s=float(times[0]-ot[0]),
        output_lift_m=float(maxz-op[0,2]),
        source_known_initial_object_xyz_m=obj[0].tolist(),
        source_known_initial_object_xyz_transformed_m=(obj[0]-np.asarray(cfg['origin_xyz_m'])).tolist(),
        source_first_to_aligned_first_lift_m=float(op[0,2]-obj[0,2]),
        output_peak_above_source_known_initial_m=float(maxz-obj[0,2]),
        output_final_error_from_source_known_initial_m=float(np.linalg.norm(op[-1]-obj[0])),
        complete_initial_lift_phase_exported=False,
        semantic_goal_lift_from_exported_first_is_15cm=False,
        output_xy_drift_m=float(np.linalg.norm(op[:,:2]-op[0,:2],axis=1).max()),
        output_object_return_error_m=float(np.linalg.norm(op[-1]-op[0])),
        peak_plateau_contiguous=bool((np.diff(plateau)==1).all()),peak_plateau_first_last=plateau[[0,-1]].tolist(),
        peak_plateau_source_times_s=times[plateau[[0,-1]]].tolist(),peak_plateau_duration_s=float((plateau[-1]-plateau[0])/fps),
        planned_withdrawal=withdrawal,actual_hand_tail=hand_end_motion,
        tail_extension_used=False,physical_phase_sync_verified=False,
        warning='Supported overlap excludes t0 object pose and about one last human recorder interval. Object starts8.262mm above the known original initial position, so retained peak lift from export start is141.738mm although peak above known source initial is150mm. Planned withdrawal completion is not certified; actual retained and dropped-tail hand displacement/speed are reported.',
        clock_map=str(ALIGNED),aligned_sha256=sha(ALIGNED))
    write_json(ROOT/'reports/stage2_alignment.json',report)
    print('ALIGNED_READY '+str(ALIGNED),flush=True)


def retarget(cfg):
    from general_motion_retargeting import GeneralMotionRetargeting,IK_CONFIG_DICT
    with np.load(ALIGNED,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    positions=data['human_joint_pos'];aa=data['human_local_axis_angle'];parents=data['joint_parents22']
    names=data['joint_names22'].tolist();n=len(positions);fps=float(data['fps'])
    matrices=Rotation.from_rotvec(aa.reshape(-1,3)).as_matrix().reshape(n,22,3,3)
    for j in range(1,22):matrices[:,j]=matrices[:,parents[j]]@matrices[:,j]
    rotations=Rotation.from_matrix(matrices.reshape(-1,3,3)).as_quat(scalar_first=True).reshape(n,22,4)
    ik_source=REPO/'GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json'
    ik=json.loads(ik_source.read_text());ik['human_scale_table']={k:1. for k in ik['human_scale_table']}
    ik_path=ROOT/'config/lift_hold_001_ik.json';write_json(ik_path,ik)
    IK_CONFIG_DICT['lift_hold_001_scene_metric']={'unitree_g1':str(ik_path)}
    origin=np.asarray(cfg['origin_xyz_m'])
    def frame(i):return {name:(positions[i,j].copy()-origin,rotations[i,j].copy()) for j,name in enumerate(names)}
    qpos_path=DATA/'baseline/lift_hold_001_g1_qpos.npy';qpos_path.parent.mkdir(parents=True,exist_ok=True)
    signature=dict(aligned_sha256=sha(ALIGNED),ik_config_sha256=sha(ik_path),robot_xml_sha256=sha(XML),urdf_sha256=sha(URDF))
    if BASELINE.exists() or qpos_path.exists():raise FileExistsError('Baseline already exists; preserve it instead of overwriting')
    start=time.monotonic()
    solver=GeneralMotionRetargeting('lift_hold_001_scene_metric','unitree_g1',actual_human_height=None,
                                   use_velocity_limit=True,verbose=False)
    assert set(solver.human_scale_table.values())=={1.}
    for _ in range(10):solver.retarget(frame(0))
    qpos=[];task_pos={};task_rot={}
    for i in range(n):
        qpos.append(solver.retarget(frame(i)))
        for label,table in [('stage1',solver.human_body_to_task1),('stage2',solver.human_body_to_task2)]:
            for key,task in table.items():
                e=task.compute_error(solver.configuration);name=label+'.'+key
                task_pos.setdefault(name,[]).append(float(np.linalg.norm(e[:3])))
                task_rot.setdefault(name,[]).append(float(np.linalg.norm(e[3:])))
        if i%60==0 or i==n-1:print(f'GMR {i+1}/{n}',flush=True)
    qpos=np.asarray(qpos);np.save(qpos_path,qpos)
    urdf=ET.parse(URDF).getroot();xml=ET.parse(XML).getroot()
    joints=[j for j in urdf.findall('joint') if j.get('type')!='fixed'];dof_names=[j.get('name') for j in joints]
    xml_names=[j.get('name') for j in xml.findall('.//body/joint')]
    bodies=[b.get('name') for w in xml.findall('worldbody') for b in w.iter('body')]
    assert len(dof_names)==len(xml_names)==29 and set(dof_names)==set(xml_names) and len(bodies)==38
    dof=qpos[:,7:][:,[xml_names.index(name) for name in dof_names]]
    lp,lr=forward_kinematics(urdf,dof,dof_names,bodies)
    rootrot=qpos[:,[4,5,6,3]].copy();rootrot/=np.linalg.norm(rootrot,axis=1,keepdims=True)
    motion=dict(fps=fps,root_pos=qpos[:,:3].copy(),root_rot=rootrot,dof_pos=dof,local_body_pos=lp,link_body_list=bodies)
    with BASELINE.open('xb') as stream:pickle.dump(motion,stream,protocol=4)
    object_path=BASELINE.with_suffix('.npz')
    data_save(object_path,trans=data['object_pos']-origin,rot=data['object_quat_wxyz'][:,[1,2,3,0]],source_time_s=data['source_time_s'])
    clearance=foot_clearance(urdf,lp,lr,bodies,motion['root_pos'],Rotation.from_quat(rootrot).as_matrix())
    limits=[]
    for j,joint in enumerate(joints):
        lim=joint.find('limit');lo=float(lim.get('lower'));hi=float(lim.get('upper'));vel=float(lim.get('velocity'))
        speed=abs(np.diff(dof[:,j])*fps)
        limits.append(dict(joint=dof_names[j],min_rad=float(dof[:,j].min()),max_rad=float(dof[:,j].max()),
            lower_rad=lo,upper_rad=hi,position_violations=int(((dof[:,j]<lo-1e-6)|(dof[:,j]>hi+1e-6)).sum()),
            max_speed_rad_s=float(speed.max()),speed_limit_rad_s=vel,speed_violations=int((speed>vel+1e-6).sum())))
    rest=trimesh.load_mesh(REST_MESH,process=False)
    op=data['object_pos']-origin;oq=data['object_quat_wxyz'][:,[1,2,3,0]]
    bottom=np.einsum('tij,vj->tvi',Rotation.from_quat(oq).as_matrix(),np.asarray(rest.vertices))[:,:,2].min(axis=1)+op[:,2]
    violations=sum(x['position_violations']+x['speed_violations'] for x in limits)
    report=dict(status='BASELINE_EXPORTED_NEEDS_STAGE4',config_sha256=sha(CONFIG),format_finite=bool(all(np.isfinite(motion[k]).all() for k in ['root_pos','root_rot','dof_pos','local_body_pos'])),
        baseline=str(BASELINE),object_pair=str(object_path),qpos=str(qpos_path),frames=n,fps=fps,
        endpoint_duration_s=(n-1)/fps,scene_uniform_scale=1.,source_origin_xyz_m=origin.tolist(),
        dof_order=dof_names,body_order=bodies,qpos_joint_order=xml_names,signature=signature,
        target_joint_position_speed_gate=violations==0,joint_limits=limits,
        ik_position_task_residual_m={k:stats(v) for k,v in task_pos.items()},
        ik_orientation_task_residual_rad={k:stats(v) for k,v in task_rot.items()},
        solver=dict(name=solver.solver,model_timestep_s=float(solver.model.opt.timestep),source_timestep_s=1/fps,
            max_extra_iterations=solver.max_iter,warm_first_pose_calls=10,velocity_limit_enabled=True,
            note='GMR executes multiple internal IK integrations per source frame; exported physical-time joint speeds are independently measured.'),
        feet=clearance_summary(clearance),object_mesh_bottom_m=stats(bottom),
        root_xy_displacement_m=float(np.linalg.norm(motion['root_pos'][:,:2]-motion['root_pos'][0,:2],axis=1).max()),
        elapsed_s=time.monotonic()-start,physical_success_verified=False,policy_or_training_run=False,
        output_sha256={str(p):sha(p) for p in [BASELINE,object_path,qpos_path]})
    write_json(ROOT/'reports/stage3_retarget.json',report)
    write_json(ROOT/'reports/stage3_qpos_cache_signature.json',dict(signature=signature,qpos_sha256=sha(qpos_path)))
    print(json.dumps(dict(status=report['status'],frames=n,joint_violations=violations,baseline=str(BASELINE),feet=report['feet'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['align','retarget','all'],default='all')
    args=parser.parse_args()
    if args.stage in ['align','all']:
        d,cfg=source_contract();align(d,cfg)
    else:cfg=json.loads(CONFIG.read_text())
    if args.stage in ['retarget','all']:
        sys.path.insert(0,str(REPO/'GMR'))
        retarget(cfg)


if __name__=='__main__':main()
