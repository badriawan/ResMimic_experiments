"""Read corrected robot/object data through the actual ResMimic HOI loader, CPU only."""
import argparse,sys,json,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,BASELINE,install,sha,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);args=p.parse_args();install();path=args.candidate.resolve()
    if ROOT not in path.parents:raise ValueError('Candidate must be inside step_2')
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
    baseline_lib=MotionLibHOI(str(BASELINE),str(BASELINE.with_suffix('.npz')),'cpu',motion_decompose=False,motion_smooth=False)
    baseline_values=baseline_lib.calc_hoi_motion_frame(ids,tt.clone())
    with BASELINE.open('rb') as f:baseline_motion=pickle.load(f)
    arm_prefixes=tuple(side+'_'+part for side in ['left','right'] for part in ['shoulder','elbow','wrist','rubber_hand'])
    inactive_body_ids=[i for i,name in enumerate(m['link_body_list']) if not name.startswith(arm_prefixes)]
    def diff_stats(actual,baseline):
        d=actual.numpy()-baseline.numpy()
        return dict(max_abs=float(np.max(np.abs(d))),rms=float(np.sqrt(np.mean(d*d))))
    conditioning_diagnostics=dict(root_linear_velocity_m_s=diff_stats(values[2],baseline_values[2]),root_angular_velocity_rad_s=diff_stats(values[3],baseline_values[3]),inactive_joint_velocity_rad_s=diff_stats(values[5][:,:15],baseline_values[5][:,:15]),inactive_local_skeleton_target_m=diff_stats(values[6][:,inactive_body_ids],baseline_values[6][:,inactive_body_ids]),note='Compared to latest step2 reference at the same480Hz. Only14armjoints are intentionally smoothed; root/inactivefeatures should remain identical. No policy success claim.',baseline_source=str(BASELINE))
    report=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,samples=sample_count,exported_fps=fps,max_errors=dict(root_position_m=rooterr,dof_rad=doferr,object_position_m=objecterr,quaternion_geodesic_rad=quaterr),numerical_tolerances=dict(position_m=2e-6,joint_rad=joint_tolerance,quaternion_geodesic_rad=2e-5,note='A-priori float32 phase/time quantization bound4eps*duration*max_joint_speed, at least1e-5rad; quaternion2e-5rad=0.00115deg'),conditioning_diagnostics=conditioning_diagnostics,quaternion_convention='XYZW on disk; actual loader outputs normalized XYZW',endpoint_query='Clamped to motion length-1e-5 to avoid native looping',input_sha256={str(p):sha(p) for p in [path,path.with_suffix('.npz'),ROOT/'scripts/foot_reference_loader.py']},simulator_executed=False,policy_executed=False,training_executed=False)
    write_json(ROOT/('reports/loader_'+path.parent.name+'.json'),report);print(json.dumps(report,indent=2));assert all(checks.values()),checks

if __name__=='__main__':main()
