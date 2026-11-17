"""Smooth object-relative lowering, preserving accepted hold and world-foot paths."""
import sys,pickle,shutil,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import ROOT,DATA,VIBRATION_DATA,install,sha,write_json
install()
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from correct_lift_hold import Chain,solve,quintic
from validate_lift_hold import load_contract
from lift_hold_validation_common import world_fk,stats


def quintic_segment(y0,v0,a0,y1,v1,a1,duration,count):
    c0=y0;c1=v0*duration;c2=a0*duration**2/2
    rhs=np.stack([y1-c0-c1-c2,v1*duration-c1-2*c2,a1*duration**2-2*c2])
    tail=np.linalg.solve(np.array([[1.,1,1],[3,4,5],[6,12,20]]),rhs)
    coeff=np.concatenate([np.stack([c0,c1,c2]),tail],axis=0)
    x=np.linspace(0,1,count)
    return sum(x[:,None]**k*coeff[k] for k in range(6))


def smooth(values,a,b,fps,sigma_s=.20,ramp_s=.60):
    """Boundary-constrained smooth target; no mixing with time-varying noise."""
    values=np.asarray(values);filtered=gaussian_filter1d(values,sigma_s*fps,axis=0,mode='nearest')
    vel=np.gradient(filtered,1/fps,axis=0);acc=np.gradient(vel,1/fps,axis=0)
    result=values.copy();result[a:b+1]=filtered[a:b+1];r=round(ramp_s*fps);duration=r/fps
    # Incoming hold curve is stationary at its endpoint; match its final discrete slope.
    v0=(values[a]-values[a-1])*fps
    result[a:a+r+1]=quintic_segment(values[a],v0,np.zeros(values.shape[1]),filtered[a+r],vel[a+r],acc[a+r],duration,r+1)
    # Use the low-frequency exit slope. Matching a single noisy outgoing
    # source interval caused overshoot; exterior poses still remain exact.
    v1=vel[b]
    result[b-r:b+1]=quintic_segment(filtered[b-r],vel[b-r],acc[b-r],values[b],v1,np.zeros(values.shape[1]),duration,r+1)
    result[a]=values[a];result[b]=values[b]
    return result


def smooth_rot(mats,a,b,fps):
    ref=Rotation.from_matrix(mats[a]);r=Rotation.from_matrix(mats)
    chart=(ref.inv()*r).as_rotvec()
    if np.linalg.norm(chart,axis=1).max()>np.pi*.9:raise ValueError('SO3 chart too close to branch cut')
    return (ref*Rotation.from_rotvec(smooth(chart,a,b,fps))).as_matrix()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',default='lowering_v2_attempt04');parser.add_argument('--projection',choices=['joint','smooth_bias'],default='smooth_bias');args=parser.parse_args()
    source=VIBRATION_DATA/'lift_hold_001_g1.pkl'
    _,old,obj,u,joints,names,bodies=load_contract(source);fps=float(old['fps']);t=np.arange(len(old['root_pos']))/fps
    decreasing=np.flatnonzero(np.diff(obj['trans'][:,2])*fps<-.0001)
    a=int(decreasing[0]);last=int(decreasing[-1]+1);b=round(11.5*fps)
    config=dict(method='Object-relative Gaussian smoothing with constrained quintic boundary segments, world-foot and arm IK',
        source=str(source),source_sha256=sha(source),object_source_sha256=sha(source.with_suffix('.npz')),
        edit_start_end_s=[float(t[a]),float(t[b])],lowering_start_end_s=[float(t[a]),float(t[last])],
        hold_core_start_end_s=[1.4166666666666667,4.916666666666667],hold_preserved_through_s=float(t[a]),
        gaussian_sigma_s=.2,boundary_segment_duration_s=.6,outgoing_velocity='filtered low-frequency slope; exterior join is not claimed C1',projection=args.projection,root_orientation_bias_rotvec_rad=[-.06,.018,-.036] if args.projection=='smooth_bias' else None,
        gates=dict(maximum_root_displacement_m=.1,maximum_root_orientation_change_deg=10.,maximum_hand_displacement_m=.1,
            maximum_hand_orientation_change_deg=35.,world_foot_position_max_error_m=.001,world_foot_rotation_max_error_deg=.5,
            high_frequency_rms_ratio_max=.2),scope='Reference kinematics only; preserve all original world foot pose trajectories and object bytes')
    write_json(ROOT/'config'/('lowering_'+args.attempt+'.json'),config)
    out=VIBRATION_DATA/'candidates'/args.attempt;out.mkdir(parents=True,exist_ok=False)
    new={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in old.items()}
    _,lr,rr,wp=world_fk(old,u,names,bodies)
    new['root_pos']=smooth(old['root_pos'],a,b,fps)
    newrr=smooth_rot(rr,a,b,fps);new['root_rot']=Rotation.from_matrix(newrr).as_quat()
    new['dof_pos']=smooth(old['dof_pos'],a,b,fps)
    lower=np.array([float(j.find('limit').get('lower')) for j in joints]);upper=np.array([float(j.find('limit').get('upper')) for j in joints])
    objr=Rotation.from_quat(obj['rot']).as_matrix();traces={}
    if args.projection=='joint':
        # The source reaches the right ankle roll limit during a swing. Allow a
        # minimal pelvis orientation correction while solving BOTH legs together,
        # rather than tilting the foot or relaxing joint limits.
        feet=['left_ankle_roll_link','right_ankle_roll_link']
        chains=[Chain(u,names,f) for f in feet];indices=np.concatenate([c.indices for c in chains]);previous_delta=np.zeros(3);joint_trace=[]
        for k in range(a+1,b):
            base_r=newrr[k].copy();reference=new['dof_pos'][k].copy()
            def residual(x):
                rotation=base_r@Rotation.from_rotvec(x[-3:]).as_matrix();q=reference.copy();q[indices]=x[:-3];rows=[]
                for chain,foot in zip(chains,feet):
                    pp,rt=chain.fk(q,new['root_pos'][k],rotation)
                    rows.extend((pp-wp[k,bodies.index(foot)])*1000)
                    rows.extend(Rotation.from_matrix((rr[k]@lr[foot][k]).T@rt).as_rotvec()*1000)
                return np.r_[rows,x[-3:]*.5,(x[-3:]-previous_delta)*.05,(x[:-3]-reference[indices])*.001]
            lo=np.r_[lower[indices]+1e-8,[-.08]*3];hi=np.r_[upper[indices]-1e-8,[.08]*3]
            x0=np.clip(np.r_[reference[indices],previous_delta],lo+1e-10,hi-1e-10)
            fit=least_squares(residual,x0,bounds=(lo,hi),max_nfev=120,ftol=1e-10,xtol=1e-10,gtol=1e-7)
            new['dof_pos'][k,indices]=fit.x[:-3];newrr[k]=base_r@Rotation.from_rotvec(fit.x[-3:]).as_matrix();previous_delta=fit.x[-3:];joint_trace.append([np.linalg.norm(fit.x[-3:]),np.max(abs(residual(fit.x)[:12]))/1000])
        new['root_rot']=Rotation.from_matrix(newrr).as_quat()
        traces['joint_leg_root_projection']=dict(orientation_adjustment_rad=stats(np.asarray(joint_trace)[:,0]),pose_residual_max_m_or_rad=stats(np.asarray(joint_trace)[:,1]))
        print('joint_leg_root_projection',traces['joint_leg_root_projection'],flush=True)
    else:
        # Smooth constant body-frame offset gives the saturated ankle room
        # without importing framewise correction noise into pelvis rotation.
        bias=np.array(config['root_orientation_bias_rotvec_rad'])
        blend=quintic((t-t[a])/.75)*quintic((t[b]-t)/.75)
        newrr=newrr@Rotation.from_rotvec(blend[:,None]*bias).as_matrix()
        new['root_rot']=Rotation.from_matrix(newrr).as_quat()
    links=['left_rubber_hand','right_rubber_hand'] if args.projection=='joint' else ['left_ankle_roll_link','right_ankle_roll_link','left_rubber_hand','right_rubber_hand']
    for link in links:
        chain=Chain(u,names,link);idx=np.array(chain.indices if 'ankle' in link else chain.indices[-7:]);targetp=wp[:,bodies.index(link)].copy();targetr=rr@lr[link]
        if 'rubber' in link:
            relative=np.einsum('tji,tj->ti',objr,targetp-obj['trans'])
            relative=smooth(relative,a,b,fps)
            targetp=obj['trans']+np.einsum('tij,tj->ti',objr,relative)
            relative_r=np.swapaxes(objr,1,2)@targetr
            targetr=objr@smooth_rot(relative_r,a,b,fps)
        previous=old['dof_pos'][a,idx].copy();trace=[]
        for k in range(a+1,b):
            reference=new['dof_pos'][k].copy()
            values,pe,re,calls=solve(chain,reference,previous,new['root_pos'][k],newrr[k],targetp[k],targetr[k],lower,upper,idx,max_nfev=150)
            new['dof_pos'][k,idx]=values;previous=values;trace.append([pe,re])
        traces[link]=dict(position_error_m=stats(np.asarray(trace)[:,0]),rotation_error_rad=stats(np.asarray(trace)[:,1]))
        print(link,traces[link],flush=True)
    exterior=(t<=t[a])|(t>=t[b])
    for key in ['root_pos','root_rot','dof_pos']:new[key][exterior]=old[key][exterior]
    new['local_body_pos']=world_fk(new,u,names,bodies)[0];new['local_body_pos'][exterior]=old['local_body_pos'][exterior]
    path=out/'lift_hold_001_g1.pkl'
    with path.open('xb') as f:pickle.dump(new,f,protocol=4)
    shutil.copyfile(source.with_suffix('.npz'),path.with_suffix('.npz'))
    write_json(ROOT/'reports'/('lowering_generation_'+args.attempt+'.json'),dict(status='PENDING_INDEPENDENT_VALIDATION',configuration=config,output=str(path),output_sha256=sha(path),script_sha256=sha(Path(__file__)),ik_target_errors=traces,source_preserved=sha(source)==config['source_sha256']))
    print(path,flush=True)

if __name__=='__main__':main()
