"""Read corrected robot/object data through the actual ResMimic HOI loader, CPU only."""
import argparse,sys,json,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,BASELINE,install,sha,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);args=p.parse_args();install();path=args.candidate.resolve()
    if ROOT not in path.parents:raise ValueError('Candidate must be inside step_3_transition')
    sys.path[:0]=[str(ROOT/'scripts')]+[str(REPO/x) for x in ['pose','legged_gym','rsl_rl','InterPose/scripts']]
    import isaacgym
    from isaacgym import gymtorch
    import torch,numpy as np
    # Import the established environment graph before NumPy2 pickle aliases.
    from baseline_config import build_configs
    from pose.utils.motion_lib_hoi import MotionLibHOI
    from foot_reference_loader import install_reference_interpolation,self_test
    from scipy.spatial.transform import Rotation,Slerp
    fixture=self_test();install_reference_interpolation()
    with path.open('rb') as f:m=pickle.load(f)
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:o={k:z[k].copy() for k in z.files}
    lib=MotionLibHOI(str(path),str(path.with_suffix('.npz')),'cpu',motion_decompose=False,motion_smooth=False)
    fps=float(m['fps']);duration=(len(m['root_pos'])-1)/fps;source_t=np.arange(len(m['root_pos']))/fps
    sample_count=int(round(duration*max(600,5*fps)))+1
    t=np.minimum(np.linspace(0,duration,sample_count),float(lib._motion_lengths[0])-1e-5)
    tt=torch.tensor(t,dtype=torch.float32);ids=torch.zeros(len(tt),dtype=torch.long)
    values=lib.calc_hoi_motion_frame(ids,tt.clone());times=tt.numpy()
    expected=lambda array:np.stack([np.interp(times,source_t,array[:,i]) for i in range(array.shape[1])],axis=1)
    rooterr=float(np.abs(values[0].numpy()-expected(m['root_pos'])).max());doferr=float(np.abs(values[4].numpy()-expected(m['dof_pos'])).max());objecterr=float(np.abs(values[9].numpy()-expected(o['trans'])).max())
    quaterr=[]
    for actual,recorded in [(values[1],m['root_rot']),(values[10],o['rot'])]:
        oracle=Slerp(source_t,Rotation.from_quat(recorded))(times);quaterr.append(float((oracle.inv()*Rotation.from_quat(actual.numpy())).magnitude().max()))
    joint_tolerance=float(max(1e-5,4*np.finfo(np.float32).eps*duration*float(np.max(np.abs(np.diff(m['dof_pos'],axis=0))*fps))))
    checks=dict(one_motion=lib.num_motions()==1,fps_matches_export=abs(float(lib._motion_fps[0])-fps)<1e-6,duration_matches=abs(float(lib._motion_lengths[0])-duration)<2e-6,all_outputs_finite=all(bool(torch.isfinite(v).all()) for v in values),robot_shape=tuple(values[4].shape)==(sample_count,29),object_shapes=tuple(values[9].shape)==(sample_count,3) and tuple(values[10].shape)==(sample_count,4),reference_adapter_fixtures=fixture['status']=='PASS',root_position_linear_agrees=rooterr<2e-6,joints_linear_agree=doferr<joint_tolerance,object_position_linear_agrees=objecterr<2e-6,root_object_slerp_agrees=max(quaterr)<2e-5)
    checks={key:bool(value) for key,value in checks.items()}
    with np.load(path.parent/'transition_clock.npz',allow_pickle=False) as z:clock={k:z[k].copy() for k in z.files}
    transition=json.loads((path.parent/'contact_plan.json').read_text())['transition'];hold=float(transition['hold_s']);start=float(transition['source_start_time_s']);h=int(round(hold*fps))
    checks['clock_control_time_matches_native']=bool(np.max(abs(clock['control_time_s']-source_t))<2e-12)
    # Check the actual cached derivative channels, not a state-only override.
    cached=dict(root_linear_velocity=lib._motion_root_vel[:h+1],root_angular_velocity=lib._motion_root_ang_vel[:h+1],joint_velocity=lib._motion_dof_vel[:h+1])
    checks['actual_loader_cached_hold_robot_velocities_zero']=all(bool((v==0).all()) for v in cached.values())
    hold_query=torch.tensor(np.arange(h+1)/fps,dtype=torch.float32);hold_ids=torch.zeros(h+1,dtype=torch.long)
    hold_values=lib.calc_hoi_motion_frame(hold_ids,hold_query.clone())
    checks['actual_loader_returned_hold_robot_velocities_zero']=all(bool((hold_values[i]==0).all()) for i in [2,3,5])
    checks['actual_loader_hold_object_cached_pose_exact']=bool((lib._object_root_pos[:h+1]==lib._object_root_pos[0]).all() and (lib._object_root_rot[:h+1]==lib._object_root_rot[0]).all())
    # MotionLibHOI returns object poses only, not object velocity. Derive these
    # diagnostics from returned poses and report float32 interpolation noise.
    op=hold_values[9].numpy();oq=hold_values[10].numpy()
    ov=np.diff(op,axis=0)*fps;oqr=Rotation.from_quat(oq);ow=(oqr[:-1].inv()*oqr[1:]).as_rotvec()*fps
    pos_roundoff_bound=8*np.finfo(np.float32).eps*max(1.,float(np.max(abs(op))))*fps
    angular_roundoff_bound=8*np.finfo(np.float32).eps*fps
    checks['actual_loader_hold_object_derived_velocity_float32_zero']=bool(np.max(abs(ov))<pos_roundoff_bound and np.max(abs(ow))<angular_roundoff_bound)
    checks['actual_loader_initial_object_derived_velocity_exact_zero']=bool(np.max(abs(ov[0]))==0 and np.max(abs(ow[0]))<1e-12)
    baseline_lib=MotionLibHOI(str(BASELINE),str(BASELINE.with_suffix('.npz')),'cpu',motion_decompose=False,motion_smooth=False)
    # Compare original baseline POSES at the shared source phase. Do not compare
    # its unscaled velocities against a held/slowed output reference.
    nt=torch.tensor(np.minimum(source_t,float(lib._motion_lengths[0])-1e-5),dtype=torch.float32);ni=torch.zeros(n:=len(nt),dtype=torch.long)
    native_values=lib.calc_hoi_motion_frame(ni,nt.clone())
    mapped_tau=np.interp(nt.numpy(),clock['control_time_s'],clock['source_reference_time_s'])
    mapped_values=baseline_lib.calc_hoi_motion_frame(ni,torch.tensor(mapped_tau,dtype=torch.float32))
    mapped_errors={key:float(np.max(abs(native_values[i].numpy()-mapped_values[i].numpy()))) for key,i in [('root_position_m',0),('joint_position_rad',4),('object_position_m',9)]}
    mapped_quat=[]
    for i in [1,10]:mapped_quat.append(float((Rotation.from_quat(native_values[i].numpy()).inv()*Rotation.from_quat(mapped_values[i].numpy())).magnitude().max()))
    checks['baseline_mapped_native_positions_agree']=mapped_errors['root_position_m']<4e-6 and mapped_errors['joint_position_rad']<2*joint_tolerance and mapped_errors['object_position_m']<4e-6
    checks['baseline_mapped_native_slerp_agrees']=max(mapped_quat)<4e-5
    bi=torch.zeros(1,dtype=torch.long);baseline_initial=baseline_lib.calc_hoi_motion_frame(bi,torch.tensor([start],dtype=torch.float32))
    conditioning_diagnostics=dict(hold_velocity_query_interval='0 <= t <= hold_s; boundary central stencil also inspected and measured exactly zero',hold_end_boundary_robot_cached_max_abs={k:float(v[h].abs().max()) for k,v in [('root_linear_velocity',lib._motion_root_vel),('root_angular_velocity',lib._motion_root_ang_vel),('joint_velocity',lib._motion_dof_vel)]},source_clock_file=str(path.parent/'transition_clock.npz'),source_start_s=start,mapped_native_pose_max_errors=mapped_errors,mapped_native_quaternion_geodesic_rad=mapped_quat,hold_robot_cached_max_abs={k:float(v.abs().max()) for k,v in cached.items()},hold_robot_returned_max_abs={k:float(hold_values[i].abs().max()) for k,i in [('root_linear',2),('root_angular',3),('joint',5)]},object_velocity_API='MotionLibHOI returns no object velocities; finite differences of actual returned object poses are used.',hold_object_derived_linear_max_component_m_s=float(abs(ov).max()),hold_object_derived_angular_max_component_rad_s=float(abs(ow).max()),object_float32_zero_tolerances=dict(linear_m_s=pos_roundoff_bound,angular_rad_s=angular_roundoff_bound),baseline_initial_loader_root_speed_m_s=float(torch.linalg.vector_norm(baseline_initial[2])),baseline_initial_loader_joint_rms_rad_s=float(torch.sqrt(torch.mean(baseline_initial[5]**2))),note='Compared positions at mapped source phase; original baseline velocities intentionally differ through hold/ramp. Frame48 is an interior source knot with central-difference baseline derivatives; cropped output frame0 is a boundary preceded by an explicit stationary hold. Baseline central derivative may differ from the source forward finite difference. No policy or physical success claim.',baseline_source=str(BASELINE))
    checks={key:bool(value) for key,value in checks.items()}
    report=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,samples=sample_count,exported_fps=fps,max_errors=dict(root_position_m=rooterr,dof_rad=doferr,object_position_m=objecterr,quaternion_geodesic_rad=quaterr),numerical_tolerances=dict(position_m=2e-6,joint_rad=joint_tolerance,quaternion_geodesic_rad=2e-5,note='A-priori float32 phase/time quantization bound4eps*duration*max_joint_speed, at least1e-5rad; quaternion2e-5rad=0.00115deg'),conditioning_diagnostics=conditioning_diagnostics,quaternion_convention='XYZW on disk; actual loader outputs normalized XYZW',endpoint_query='Clamped to motion length-1e-5 to avoid native looping',input_sha256={str(p):sha(p) for p in [path,path.with_suffix('.npz'),ROOT/'scripts/foot_reference_loader.py',path.parent/'transition_clock.npz',BASELINE,BASELINE.with_suffix('.npz')]},simulator_executed=False,policy_executed=False,training_executed=False)
    write_json(ROOT/('reports/loader_'+path.parent.name+'.json'),report);print(json.dumps(report,indent=2));assert all(checks.values()),checks

if __name__=='__main__':main()
