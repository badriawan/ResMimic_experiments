"""Build a shared robot/object rest-hold and smooth source-clock ramp.

This changes time consistently, not independently zeroing reset velocities.
No simulator, controller, training, or dynamic-balance claim is executed here.
"""
import argparse,copy,json,pickle,sys,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,BASELINE,ASSETS,URDF,install,sha,write_json

def audit_initial(motion,objects,robot,joint_names):
    import numpy as np
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    from convert_gmr_qpos import forward_kinematics,foot_clearance,origin
    fps=float(motion['fps']);count=min(len(motion['root_pos']),int(round(.8*fps))+1)
    q=motion['dof_pos'][:count];rp=motion['root_pos'][:count];rq=motion['root_rot'][:count];rr=Rotation.from_quat(rq).as_matrix()
    local,rot=forward_kinematics(robot,q,joint_names,motion['link_body_list']);body_names=list(motion['link_body_list'])
    world=rp[:,None,:]+np.einsum('tij,tkj->tki',rr,local);feet=foot_clearance(robot,local,rot,body_names,rp,rr)
    total=0.;com=np.zeros((count,3))
    for link in robot.findall('link'):
        inertial=link.find('inertial')
        if inertial is None:continue
        mass=float(inertial.find('mass').get('value'));position,_=origin(inertial);index=body_names.index(link.get('name'))
        com+=mass*(world[:,index]+np.einsum('tij,j->ti',rr@rot[link.get('name')],position));total+=mass
    com/=total
    rootv=np.gradient(rp,1/fps,axis=0);qd=np.gradient(q,1/fps,axis=0);objectv=np.gradient(objects['trans'][:count],1/fps,axis=0)
    # A near-floor band of the two exact capsule primitives, not measured support forces.
    # Sphere-end samples include exact world-down support point; horizontal-circle
    # samples close to the floor provide a conservative sampled support footprint.
    directions=[]
    for gamma in np.linspace(0,.9,19):
        for phi in np.linspace(0,2*np.pi,48,endpoint=False):directions.append([np.sin(gamma)*np.cos(phi),np.sin(gamma)*np.sin(phi),-np.cos(gamma)])
    directions=np.asarray(directions)
    rows=[]
    for frame in range(count):
        support=[];footdata={}
        for foot in ['left_ankle_roll_link','right_ankle_roll_link']:
            bid=body_names.index(foot);wr=rr[frame]@rot[foot][frame];wp=world[frame,bid];gap=float(feet[foot]['capsule_surrogate'][frame]);points=[]
            for collision in robot.find("link[@name='%s']"%foot).findall('collision'):
                xyz,r=origin(collision);geometry=list(collision.find('geometry'))[0]
                if geometry.tag!='cylinder':continue
                axis=(wr@r)[:,2];center=wp+wr@xyz;radius=float(geometry.get('radius'));half=float(geometry.get('length'))/2
                samples=np.concatenate([(center+sign*half*axis)[None]+radius*directions for sign in [-1,1]])
                points.extend(samples[samples[:,2]<=.005+1e-12])
            points=np.asarray(points).reshape(-1,3);footdata[foot]=dict(capsule_floor_gap_m=gap,near_floor_point_count=len(points))
            if gap<=.005 and len(points):support.extend(points[:,:2])
        support=np.asarray(support).reshape(-1,2)
        margin=None;inside=False
        if len(support)>=3:
            hull=ConvexHull(support);margin=float(-np.max(hull.equations[:,:2]@com[frame,:2]+hull.equations[:,2]));inside=margin>=0
        rows.append(dict(frame=frame,time_s=frame/fps,root_speed_m_s=float(np.linalg.norm(rootv[frame])),joint_speed_rms_rad_s=float(np.sqrt(np.mean(qd[frame]**2))),joint_speed_max_rad_s=float(abs(qd[frame]).max()),object_speed_m_s=float(np.linalg.norm(objectv[frame])),robot_com_world_m=com[frame].tolist(),sampled_near_floor_support_margin_m=margin,robot_com_inside_sampled_support=inside,feet=footdata))
    # The initial hold removes the need to enter with this source velocity, while
    # preserving a pose whose geometric support proxy can be independently checked.
    report=dict(status='INITIAL_GEOMETRY_AND_KINEMATIC_AUDIT',reference=str(BASELINE),frames_examined=count,fps=fps,robot_mass_kg=total,initial=rows[0],rows=rows,interpretation=['Capsule geometry uses the configured cylinder-to-capsule surrogate.','Near-floor samples use a 5 mm band and are geometric proxies, not measured contact forces or ZMP/CoP.','Robot COM excludes the floor-supported lamp; physical grip forces and combined carrying balance are not certified.','A rest-hold is only a consistent reference initialization, not proof the policy will hold it.'])
    write_json(ROOT/'reports/initial_audit.json',report)
    return report

def source_clock(source_count,fps,hold,ramp):
    import numpy as np
    hold_steps=int(round(hold*fps));ramp_steps=int(round(ramp*fps))
    if hold_steps<2 or ramp_steps<4 or abs(hold_steps/fps-hold)>1e-12 or abs(ramp_steps/fps-ramp)>1e-12:raise ValueError('Hold/ramp must align to native frames, hold>=2frames')
    if ramp_steps%2:raise ValueError('Ramp steps must be even so source join is a native knot')
    source_join=ramp_steps//2
    if source_join>=source_count-1:raise ValueError('Ramp consumes entire source clip')
    total_steps=hold_steps+ramp_steps+(source_count-1-source_join);time=np.arange(total_steps+1)/fps
    tau=np.zeros_like(time);rate=np.zeros_like(time);acc=np.zeros_like(time);jerk=np.zeros_like(time);phase=np.zeros(len(time),dtype=np.int8)
    first=hold_steps;last=hold_steps+ramp_steps;u=np.arange(ramp_steps+1)/ramp_steps
    tau[first:last+1]=ramp*(2.5*u**4-3*u**5+u**6)
    rate[first:last+1]=10*u**3-15*u**4+6*u**5
    acc[first:last+1]=(30*u**2-60*u**3+30*u**4)/ramp
    jerk[first:last+1]=(60*u-180*u**2+120*u**3)/ramp**2
    phase[first:last]=1
    tau[last:]=np.arange(source_join,source_count)/fps;rate[last:]=1;phase[last:]=2
    return dict(control_time_s=time,source_reference_time_s=tau,clock_rate=rate,clock_acceleration_s_inv=acc,clock_jerk_s_inv2=jerk,phase_code=phase),dict(hold_frames=hold_steps,ramp_frames=ramp_steps,join_output_frame=last,join_source_frame=source_join)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--name',default='attempt01');parser.add_argument('--hold',type=float,default=.75);parser.add_argument('--ramp',type=float,default=1.);parser.add_argument('--start-frame',type=int,default=0);parser.add_argument('--audit-only',action='store_true');args=parser.parse_args();install()
    import numpy as np
    from scipy.spatial.transform import Rotation,Slerp
    sys.path.insert(1,str(REPO/'legged_gym/legged_gym/scripts'))
    from convert_gmr_qpos import forward_kinematics
    with BASELINE.open('rb') as f:baseline=pickle.load(f)
    with np.load(BASELINE.with_suffix('.npz'),allow_pickle=False) as z:objects={k:z[k].copy() for k in z.files}
    robot=ET.parse(URDF).getroot();joints=[j for j in robot.findall('joint') if j.get('type') in ('revolute','continuous')];names=[j.get('name') for j in joints]
    audit=audit_initial(baseline,objects,robot,names)
    print('Initial audit',json.dumps(audit['initial']),flush=True)
    if args.audit_only:return
    if args.start_frame < 0 or args.start_frame >= audit['frames_examined']:raise ValueError('Start frame must be within audited initial window')
    initial=audit['rows'][args.start_frame]
    if not initial['robot_com_inside_sampled_support']:raise ValueError('Initial robot COM is outside sampled support proxy; choose/review a different initial pose')
    if not args.name.replace('_','').replace('-','').isalnum():raise ValueError('Invalid candidate name')
    out=ROOT/'candidates'/args.name;out.mkdir(parents=True,exist_ok=False)
    fps=float(baseline['fps']);clock,indices=source_clock(len(baseline['root_pos'])-args.start_frame,fps,args.hold,args.ramp);clock['source_reference_time_s']+=args.start_frame/fps;t=clock['source_reference_time_s'];native=np.arange(len(baseline['root_pos']))/fps
    indices['join_source_frame']+=args.start_frame;indices['start_source_frame']=args.start_frame
    # Restore original native grid exactly; avoid offset addition roundoff.
    t[indices['join_output_frame']:]=native[indices['join_source_frame']:]
    interp=lambda a:np.stack([np.interp(t,native,a[:,i]) for i in range(a.shape[1])],axis=1)
    result=copy.deepcopy(baseline);result['root_pos']=interp(baseline['root_pos']);result['dof_pos']=interp(baseline['dof_pos']);result['root_rot']=Slerp(native,Rotation.from_quat(baseline['root_rot']))(t).as_quat()
    ob=dict(trans=interp(objects['trans']),rot=Slerp(native,Rotation.from_quat(objects['rot']))(t).as_quat(),source_time_s=np.interp(t,native,objects['source_time_s']))
    # Exact repeated initial state and exact source-native suffix, no sign-roundtrip.
    hold_end=indices['hold_frames'];join_out=indices['join_output_frame'];join_source=indices['join_source_frame']
    for key in ['root_pos','root_rot','dof_pos']:
        result[key][:hold_end+1]=baseline[key][args.start_frame];result[key][join_out:]=baseline[key][join_source:]
    for key in ['trans','rot','source_time_s']:
        ob[key][:hold_end+1]=objects[key][args.start_frame];ob[key][join_out:]=objects[key][join_source:]
    result['local_body_pos'],_=forward_kinematics(robot,result['dof_pos'],names,result['link_body_list'])
    result['local_body_pos'][:hold_end+1]=baseline['local_body_pos'][args.start_frame];result['local_body_pos'][join_out:]=baseline['local_body_pos'][join_source:]
    filename=out/'interpose_floorlamp_g1_transition.pkl'
    with filename.open('xb') as f:pickle.dump(result,f,protocol=4)
    np.savez_compressed(filename.with_suffix('.npz'),**ob);np.savez_compressed(out/'transition_clock.npz',**clock)
    plan=json.loads((BASELINE.parent/'contact_plan.json').read_text());plan['contact_start_frame']=0;plan['hold_start_frame']=0;plan['contact_end_frame']=len(t)-1;plan['contact_phase_frames']=[0,len(t)-1]
    if 'frames' in plan:plan['frames']=len(t)
    plan['fps']=fps;plan['transition']=dict(method='Shared source clock: exact rest hold, integrated quintic speed ramp, original 1x suffix',hold_s=args.hold,ramp_s=args.ramp,source_start_time_s=args.start_frame/fps,source_join_time_s=join_source/fps,output_join_time_s=join_out/fps,source_clock_file='transition_clock.npz',pregrasped=True,initial_velocity_expected='Zero for all robot/object reference channels during hold; loader derivatives recomputed')
    write_json(out/'contact_plan.json',plan)
    qvel=np.diff(result['dof_pos'],axis=0)*fps;rv=np.diff(result['root_pos'],axis=0)*fps;ov=np.diff(ob['trans'],axis=0)*fps
    quaternion_hold_exact=all(np.array_equal(a[:hold_end+1],np.broadcast_to(a[0],a[:hold_end+1].shape)) for a in [result['root_rot'],ob['rot']])
    speeds=np.asarray([float(j.find('limit').get('velocity')) for j in joints]);lo=np.asarray([float(j.find('limit').get('lower')) for j in joints]);hi=np.asarray([float(j.find('limit').get('upper')) for j in joints])
    suffix_pose_exact=all(np.array_equal(result[key][join_out:],baseline[key][join_source:]) for key in ['root_pos','root_rot','dof_pos','local_body_pos'])
    suffix_object_exact=all(np.array_equal(ob[key][join_out:],objects[key][join_source:]) for key in ob)
    checks=dict(all_finite=all(np.isfinite(v).all() for v in result.values() if isinstance(v,np.ndarray)) and all(np.isfinite(v).all() for v in ob.values()),unit_quaternions=max(float(np.max(abs(np.linalg.norm(result['root_rot'],axis=1)-1))),float(np.max(abs(np.linalg.norm(ob['rot'],axis=1)-1))))<1e-6,source_clock_monotone=bool((np.diff(t)>=-1e-12).all()),clock_rate_zero_to_one=bool((clock['clock_rate']>=-1e-12).all() and (clock['clock_rate']<=1+1e-12).all()),requested_source_interval_retained=bool(t[0]==args.start_frame/fps and t[-1]==native[-1]),suffix_robot_exact=suffix_pose_exact,suffix_object_exact=suffix_object_exact,initial_all_velocity_zero=bool(np.max(abs(qvel[:hold_end]))==0 and np.max(abs(rv[:hold_end]))==0 and np.max(abs(ov[:hold_end]))==0 and quaternion_hold_exact),joint_limits=bool(((result['dof_pos']>=lo-1e-8)&(result['dof_pos']<=hi+1e-8)).all()),joint_velocity_95pct=bool((abs(qvel)/speeds).max()<=.95+1e-7),initial_support_proxy_inside=bool(initial['robot_com_inside_sampled_support']))
    report=dict(status='PASS_TRANSITION_BUILDER_PENDING_INDEPENDENT_VALIDATION' if all(checks.values()) else 'FAIL_TRANSITION_BUILDER',checks=checks,frames=len(t),fps=fps,output_duration_s=clock['control_time_s'][-1],source_duration_s=native[-1],arguments=vars(args),indices=indices,initial_source_state=initial,discarded_source_prefix_frames=args.start_frame,source_time_s_semantics='Object source_time_s remains the interpolated original capture clock; source_reference_time_s is relative to original 480Hz reference frame0.',mapping=dict(time='tau=start_frame/fps during hold; ramp tau=start_frame/fps+T*(2.5u^4-3u^5+u^6); suffix tau=start_frame/fps+.5T+(t-hold-T)',rate='10u^3-15u^4+6u^5; range0..1',acceleration='30u^2(1-u)^2/T',jerk='60u(1-u)(1-2u)/T^2',join_continuity='Clock position/rate/acceleration/jerk match at both joins. Original piecewise pose path does not become globally C3.'),maximum_source_clock_acceleration_s_inv=float(clock['clock_acceleration_s_inv'].max()),initial_output_root_speed_m_s=float(np.linalg.norm(rv[0])),initial_output_joint_velocity_rms_rad_s=float(np.sqrt(np.mean(qvel[0]**2))),initial_output_object_speed_m_s=float(np.linalg.norm(ov[0])),maximum_joint_velocity_ratio=float((abs(qvel)/speeds).max()),source_sha256={str(p):sha(p) for p in [BASELINE,BASELINE.with_suffix('.npz'),BASELINE.parent/'contact_plan.json',URDF]},outputs_sha256={str(p):sha(p) for p in [filename,filename.with_suffix('.npz'),out/'transition_clock.npz',out/'contact_plan.json']},limitations=['Initial COM/near-floor footprint is a geometric proxy, not measured stable support or whole-body balance.','Pregrasped object starts above floor by its existing clearance; holding/preparation adds no external force or constraint.','Shared time mapping retains the original source geometry; new sample interpolation still needs independent dense validation.','Initial robot and object velocities are consistent reference zeros because all poses are repeated, not state-only velocity overrides.','No policy, physics simulation, or training is executed by this builder.'])
    write_json(out/'builder_report.json',report);write_json(ROOT/('reports/builder_'+args.name+'.json'),report);print(json.dumps(report,indent=2),flush=True)
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
