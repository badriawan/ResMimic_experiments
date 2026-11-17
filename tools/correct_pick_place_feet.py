"""Precision source-informed support/swing reference; no physics or policy.

Build a new candidate, retaining old attempts and exact successful upper-body IK.
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
from foot_contact_support import (WORK, REPO, np, Rotation, Slerp, Chain, solve_chain,
                                  phase, evaluate, forward_kinematics, foot_clearance,
                                  sole_offset, sha, stats, FEET, origin, runs)

# Endpoints are planted; the open intervals between them are swing.
# One full frame of double support separates successive swings.
SWINGS = [[(34,48),(69,87),(104,121)],[(49,67),(88,103)]]


def quintic(u):
    u=np.clip(u,0.,1.)
    return u**3*(10+u*(-15+6*u))


def interpolate_motion(m, subdivision=20):
    n=len(m['root_pos'])
    t=np.arange((n-1)*subdivision+1,dtype=float)/subdivision
    lo=np.minimum(t.astype(int),n-2)
    w=(t-lo)[:,None]
    dense=dict(m)
    dense['root_pos']=(1-w)*m['root_pos'][lo]+w*m['root_pos'][lo+1]
    dense['dof_pos']=(1-w)*m['dof_pos'][lo]+w*m['dof_pos'][lo+1]
    dense['root_rot']=Slerp(np.arange(n),Rotation.from_quat(m['root_rot']))(t).as_quat()
    return dense,t


def dense_assessment(m,urdf,names,subdivision=20):
    dense,t=interpolate_motion(m,subdivision)
    lp,lr=forward_kinematics(urdf,dense['dof_pos'],names,m['link_body_list'])
    rr=Rotation.from_quat(dense['root_rot']).as_matrix()
    world=dense['root_pos'][:,None]+np.einsum('tij,tbj->tbi',rr,lp)
    clear=foot_clearance(urdf,lp,lr,m['link_body_list'],dense['root_pos'],rr)
    dt=1/(m['fps']*subdivision)
    near_motion={}
    point_motion={}
    variants=['cylinder','capsule_surrogate']
    support=np.zeros((len(t),2),bool)
    for s,foot in enumerate(FEET):
        xyz=world[:,m['link_body_list'].index(foot)]
        speed=np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1)/dt
        gap=np.minimum(clear[foot]['cylinder'],clear[foot]['capsule_surrogate'])
        near=(gap[:-1]<.005)|(gap[1:]<.005)
        near_motion[foot]=dict(samples=int(near.sum()),speed_m_s=stats(speed[near]),
                               above_0_1m_s=int((near&(speed>.1)).sum()))
        wr=rr@lr[foot]
        # Track material centres/endpoints of every foot capsule. This catches
        # yaw rotation near the floor that ankle-origin velocity alone misses.
        speeds=[]
        for collision in urdf.find("link[@name='%s']"%foot).findall('collision'):
            cpos,crot=origin(collision)
            geom=list(collision.find('geometry'))[0]
            half=float(geom.get('length'))/2
            radius=float(geom.get('radius'))
            for sign in [-1,0,1]:
                local=cpos+sign*half*crot[:,2]
                point=xyz+np.einsum('tij,j->ti',wr,local)
                point_gap=point[:,2]-radius
                pn=(point_gap[:-1]<.005)|(point_gap[1:]<.005)
                ps=np.linalg.norm(np.diff(point[:,:2],axis=0),axis=1)/dt
                speeds.extend(ps[pn])
        point_motion[foot]=dict(near_floor_probe_speed_m_s=stats(speeds),
                                above_0_1m_s=int((np.array(speeds)>.1).sum()))
        support[:,s]=np.maximum(abs(clear[foot]['cylinder']),abs(clear[foot]['capsule_surrogate']))<=.005
    # All checks independent of source stance masks.
    min_gap=min(v.min() for f in clear.values() for v in f.values())
    no_support=np.flatnonzero(~support.any(axis=1))
    assessment=dict(samples=len(t),sample_rate_hz=m['fps']*subdivision,
                    minimum_gap_m=float(min_gap),unsupported_sample_count=len(no_support),
                    unsupported_times_s=(t[no_support]/m['fps']).tolist(),
                    near_floor_definition='Either endpoint gap <5mm, native cylinder OR capsule; all intervals without phase filtering',
                    near_floor_ankle_motion=near_motion,near_floor_capsule_probes=point_motion,
                    finite_subframe_sampling=True)
    return assessment,world,clear


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--motion',type=Path,required=True)
    parser.add_argument('--aligned',type=Path,required=True)
    parser.add_argument('--upper-seed',type=Path,default=WORK/'inputs/pick_place_001/upper_body_seed.pkl')
    parser.add_argument('--object-mesh',type=Path)
    parser.add_argument('--swing-height',type=float,default=.04)
    parser.add_argument('--vertical-fraction',type=float,default=.20)
    parser.add_argument('--subdivision',type=int,default=4)
    parser.add_argument('--initial-right-xy',type=float,nargs=2,default=[.07,.145])
    parser.add_argument('--initial-right-yaw',type=float,default=3.4)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if WORK not in out.parents or out.exists():
        raise ValueError('Use a new output directory inside this custom-data repository')
    if not .025<=args.swing_height<=.08 or not .1<=args.vertical_fraction<=.3:
        raise ValueError('Unreviewed swing parameters')
    if args.subdivision not in [1,2,4,8]:
        raise ValueError('Unreviewed precision grid')
    out.mkdir(parents=True)
    start=time.monotonic()
    original=args.motion.resolve(strict=True)
    prior=args.upper_seed.resolve(strict=True)
    source_path=args.aligned.resolve(strict=True)
    urdf_path=REPO/'assets/g1/g1_custom_collision_29dof.urdf'
    mesh=args.object_mesh.resolve(strict=True) if args.object_mesh else original.parent/'floorlamp.obj'
    inputs=[original,original.with_suffix('.npz'),prior,source_path,urdf_path,mesh,Path(__file__).with_name('foot_contact_support.py'),Path(__file__).resolve()]
    hashes={str(p):sha(p) for p in inputs}
    with original.open('rb') as f: baseline_sparse=pickle.load(f)
    with prior.open('rb') as f: upper_sparse=pickle.load(f)
    urdf=ET.parse(urdf_path).getroot()
    joints=[j for j in urdf.findall('joint') if j.get('type')!='fixed']
    names=[j.get('name') for j in joints]
    lower=np.array([float(j.find('limit').get('lower')) for j in joints])
    higher=np.array([float(j.find('limit').get('upper')) for j in joints])
    limits=np.array([float(j.find('limit').get('velocity')) for j in joints])
    # Avoid the hyperextended-knee branch (URDF permits a small negative angle).
    # This is an additional posture constraint, never a relaxed URDF limit.
    posture_lower=lower.copy()
    for side in ['left','right']:
        posture_lower[names.index(side+'_knee_joint')]=.02
    with np.load(source_path,allow_pickle=False) as data:
        markers=data['human_joint_pos'][:,[10,11]]*.9
    phases=[phase(markers[:,s],30) for s in range(2)]
    seeds=np.stack([p[0] for p in phases],axis=1)
    n_sparse=len(markers)
    if n_sparse!=126 or baseline_sparse['fps']!=30:raise ValueError('This correction is specific to the bundled 126-frame pick_place_001 clip')
    stance_sparse=np.ones((n_sparse,2),bool)
    for s in range(2):
        for a,b in SWINGS[s]: stance_sparse[a+1:b,s]=False
    _,world_sparse,lr_sparse,_=evaluate(baseline_sparse,urdf,names,stance_sparse)
    subdivision=args.subdivision
    baseline,tt=interpolate_motion(baseline_sparse,subdivision)
    upper,_=interpolate_motion(upper_sparse,subdivision)
    for data in [baseline,upper]:
        data['fps']=30*subdivision
        data['local_body_pos'],_=forward_kinematics(urdf,data['dof_pos'],names,data['link_body_list'])
    m={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in upper.items()}
    n=len(tt);fps=m['fps']
    stance=np.ones((n,2),bool)
    for s in range(2):
        for a,b in SWINGS[s]: stance[(tt>a)&(tt<b),s]=False
    assert stance.any(axis=1).all()
    before,world0,lr0,_=evaluate(baseline,urdf,names,stance)
    rr=Rotation.from_quat(m['root_rot']).as_matrix()
    positions=[];rotations=[];anchor_report={}
    for s,foot in enumerate(FEET):
        foot_idx=m['link_body_list'].index(foot)
        wp=world_sparse[:,foot_idx]
        wr=Rotation.from_quat(baseline_sparse['root_rot']).as_matrix()@lr_sparse[foot]
        yaw=np.unwrap(np.arctan2(wr[:,1,0],wr[:,0,0]))
        anchors=[]
        target_p=np.zeros((n,3));target_yaw=np.zeros(n)
        for a,b in runs(stance_sparse[:,s]):
            selected=np.flatnonzero(seeds[a:b,s])+a
            method='original high-confidence low-height/low-speed core'
            if len(selected)<3:
                # Last L landing has no confident seed in v1. Use the slowest
                # local source frames, explicitly recording this extrapolation.
                local_speed=phases[s][1][a:b]
                selected=np.arange(a,b)[local_speed<=np.percentile(local_speed,50)]
                method='local slowest-half source frames; inferred terminal landing'
            p=np.median(wp[selected],axis=0)
            y=float(np.median(yaw[selected]))
            source_p=p.copy();source_y=y
            if s==1 and a==0:
                p[:2]=args.initial_right_xy
                y=args.initial_right_yaw
                method+='; constant footprint/heading corrected for full-stance URDF reach'
            r=Rotation.from_rotvec([0,0,y]).as_matrix()
            p[2]=.0005-sole_offset(urdf,foot,r)
            sel=(tt>=a)&(tt<=b-1)
            target_p[sel]=p
            target_yaw[sel]=y
            anchors.append(dict(start_frame=int(a),stop_frame_exclusive=int(b),
                                selected_source_frames=selected.tolist(),method=method,
                                position_m=p.tolist(),yaw_rad=y,
                                source_anchor_position_m=source_p.tolist(),source_anchor_yaw_rad=source_y,
                                xy_adjustment_m=float(np.linalg.norm(p[:2]-source_p[:2])),
                                yaw_adjustment_deg=float(np.degrees(y-source_y))))
        for a,b in SWINGS[s]:
            ia,ib=a*subdivision,b*subdivision
            u=(tt[ia:ib+1]-a)/(b-a)
            fraction=args.vertical_fraction
            weight=quintic((u-fraction)/(1-2*fraction))
            target_p[ia:ib+1]=(1-weight[:,None])*target_p[ia]+weight[:,None]*target_p[ib]
            if s==1 and a==49:
                # Preserve the source heading progression in this large turn.
                # Independent delayed/early turns hit opposite ankle-roll
                # limits. Blend to fixed endpoint headings while safely aloft.
                ya,yb=target_yaw[ia],target_yaw[ib]
                curve=np.interp(tt[ia:ib+1],np.arange(n_sparse),yaw)
                curve+=(1-quintic(u))*(ya-yaw[a])+quintic(u)*(yb-yaw[b])
                begin=quintic((u-.055)/.08)
                end=quintic((u-.72)/.08)
                curve=(1-begin)*ya+begin*curve
                target_yaw[ia:ib+1]=(1-end)*curve+end*yb
            else:
                target_yaw[ia:ib+1]=(1-weight)*target_yaw[ia]+weight*target_yaw[ib]
            lift=np.where(u<fraction,quintic(u/fraction),
                          np.where(u>1-fraction,quintic((1-u)/fraction),1.))
            target_p[ia:ib+1,2]+=args.swing_height*lift
        target_r=Rotation.from_rotvec(np.column_stack([np.zeros(n),np.zeros(n),target_yaw])).as_matrix()
        positions.append(target_p);rotations.append(target_r);anchor_report[foot]=anchors
    failures=[];residuals={};orientation_residuals={};nfev=0;restarts=[];traces={}
    for s,foot in enumerate(FEET):
        chain=Chain(urdf,names,foot)
        indices=np.array(chain.indices)
        previous=m['dof_pos'][0,indices]
        residual=[];rotation_residual=[];trace=[]
        for t in range(n):
            q,success,count=solve_chain(chain,baseline['dof_pos'][t],previous,m['root_pos'][t],rr[t],
                                        positions[s][t],rotations[s][t],posture_lower,higher,indices)
            chosen_seed='previous'
            def error(values):
                test=baseline['dof_pos'][t].copy();test[indices]=values
                p,r=chain.fk(test,m['root_pos'][t],rr[t])
                pe=float(np.linalg.norm(p-positions[s][t]))
                re=float(Rotation.from_matrix(rotations[s][t].T@r).magnitude())
                return pe,re,pe*1000+re*100
            pe,re,score=error(q)
            if pe>.0001 or re>.001:
                restart_seeds=[baseline['dof_pos'][t,indices].copy()]
                bent=restart_seeds[0].copy();bent[3]=1.
                restart_seeds.append(bent)
                for seed_id,seed in enumerate(restart_seeds):
                    alternative,ok,more=solve_chain(chain,baseline['dof_pos'][t],seed,m['root_pos'][t],rr[t],
                                             positions[s][t],rotations[s][t],posture_lower,higher,indices)
                    count+=more
                    ape,are,ascore=error(alternative)
                    if ascore<score:
                        q,success,pe,re,score=alternative,ok,ape,are,ascore
                        chosen_seed=['baseline','flexed_knee'][seed_id]
                    if pe<=.0001 and re<=.001:break
                restarts.append(dict(foot=foot,output_frame=t,position_error_m=pe,orientation_error_rad=re))
            m['dof_pos'][t,indices]=q;previous=q;nfev+=count
            p,r=chain.fk(m['dof_pos'][t],m['root_pos'][t],rr[t])
            residual.append(np.linalg.norm(p-positions[s][t]))
            rotation_residual.append(float(np.degrees(Rotation.from_matrix(rotations[s][t].T@r).magnitude())))
            trace.append(dict(output_frame=t,source_frame=float(tt[t]),seed=chosen_seed,
                              position_error_m=pe,orientation_error_rad=re,converged=bool(success)))
            if not success: failures.append(dict(foot=foot,frame=t))
        residuals[foot]=stats(residual)
        orientation_residuals[foot]=stats(rotation_residual);traces[foot]=trace
        print('Solved',foot,'max target residual',max(residual),flush=True)
    m['local_body_pos'],_=forward_kinematics(urdf,m['dof_pos'],names,m['link_body_list'])
    after,world1,lr1,clearance=evaluate(m,urdf,names,stance)
    dense,dense_world,dense_clear=dense_assessment(m,urdf,names,subdivision=max(1,600//fps))
    speeds=abs(np.diff(m['dof_pos'],axis=0))*fps
    hands={}
    for side in ['left','right']:
        for suffix in ['wrist_yaw_link','rubber_hand']:
            name=side+'_'+suffix;i=m['link_body_list'].index(name)
            hands[name]=dict(position_change_m=stats(np.linalg.norm(world1[:,i]-world0[:,i],axis=1)),
                            orientation_change_deg=stats(np.degrees(Rotation.from_matrix(np.swapaxes(lr0[name],1,2)@lr1[name]).magnitude())))
    # No hiding sliding by excluding unknown frames: dense near-ground gates
    # use actual FK heights, plus a no-flight gate independent of phase labels.
    gates=dict(target_residual_5mm=all(r['max']<=.005 for r in residuals.values()),
               target_orientation_1deg=all(r['max']<=1 for r in orientation_residuals.values()),
               stance_gap_5mm=all(abs(v[stance[:,s]]).max()<=.005 for s,f in enumerate(FEET) for v in clearance[f].values()),
               stance_drift_10mm=all(seg['max_drift_from_first_m']<=.01 for f in after['feet'].values() for seg in f['segments']),
               stance_speed_p95_5cm_s=all(f['stance_speed_m_s']['p95']<=.05 for f in after['feet'].values()),
               dense_penetration_2mm=dense['minimum_gap_m']>=-.002,
               dense_support_all_samples=dense['unsupported_sample_count']==0,
               all_near_floor_ankle_speed_10cm_s=all(v['speed_m_s']['max']<=.1 for v in dense['near_floor_ankle_motion'].values()),
               all_near_floor_ankle_speed_p95_5cm_s=all(v['speed_m_s']['p95']<=.05 for v in dense['near_floor_ankle_motion'].values()),
               all_near_floor_capsule_probe_speed_10cm_s=all(v['near_floor_probe_speed_m_s']['max']<=.1 for v in dense['near_floor_capsule_probes'].values()),
               all_near_floor_capsule_probe_speed_p95_5cm_s=all(v['near_floor_probe_speed_m_s']['p95']<=.05 for v in dense['near_floor_capsule_probes'].values()),
               position_limits=bool(((m['dof_pos']>=lower-1e-6)&(m['dof_pos']<=higher+1e-6)).all()),
               speed_limits=bool((speeds<=limits+1e-6).all()),
               hand_position_5mm=all(h['position_change_m']['max']<=.005 for h in hands.values()),
               hand_orientation_1deg=all(h['orientation_change_deg']['max']<=1 for h in hands.values()),
               finite=all(np.isfinite(v).all() for v in m.values() if isinstance(v,np.ndarray)),
               solver_converged=not failures, inputs_preserved=all(sha(Path(p))==h for p,h in hashes.items()),
               positive_knee_posture=bool((m['dof_pos'][:,[names.index('left_knee_joint'),names.index('right_knee_joint')]]>=.02-1e-8).all()),
               reused_upper_body_exact=bool(np.array_equal(m['dof_pos'][:,12:],upper['dof_pos'][:,12:])
                         and np.array_equal(m['root_pos'],upper['root_pos']) and np.array_equal(m['root_rot'],upper['root_rot'])))
    motion=out/'interpose_floorlamp_g1_foot_contact.pkl'
    with motion.open('xb') as f:pickle.dump(m,f,protocol=4)
    with np.load(original.with_suffix('.npz'),allow_pickle=False) as obj:
        object_sparse={key:obj[key].copy() for key in obj.files}
    trans=np.stack([np.interp(tt,np.arange(n_sparse),object_sparse['trans'][:,i]) for i in range(3)],axis=1)
    rot=Slerp(np.arange(n_sparse),Rotation.from_quat(object_sparse['rot']))(tt).as_quat()
    rot[::subdivision]=object_sparse['rot']
    times=np.interp(tt,np.arange(n_sparse),object_sparse['source_time_s'])
    np.savez_compressed(motion.with_suffix('.npz'),trans=trans,rot=rot,source_time_s=times)
    gates['object_original_knots_exact']=bool(np.array_equal(trans[::subdivision],object_sparse['trans'])
                                  and np.array_equal(rot[::subdivision],object_sparse['rot'])
                                  and np.array_equal(times[::subdivision],object_sparse['source_time_s']))
    gates['same_endpoint_duration']=bool(abs((n-1)/fps-(n_sparse-1)/30)<1e-12)
    report=dict(status='PASS_REFERENCE_KINEMATIC' if all(gates.values()) else 'FAIL_REFERENCE_KINEMATIC',
                scope='Priority1 full kinematic reference contact schedule only; inferred, not measured physical contacts',
                frames=n,fps=fps,gates=gates,source_schedule=dict(swing_endpoint_frames_at_source_30Hz=SWINGS,
                meaning='Source-informed reconstructed walking reference with non-flight assumption; not recovered force labels',
                one_frame_double_support_handoffs=True,stance_anchors=anchor_report,
                vertical_swing_fraction=args.vertical_fraction,swing_clearance_m=args.swing_height,
                first_right_swing_yaw='Original source heading curve with endpoint corrections; fixed until fraction .055, blended by .135, fixed from .8',
                interpolation='Quintic vertical rise, then horizontal/yaw translation, then fixed-XY vertical landing',
                prior_unknown_frames_reconstructed=True),
                source_seed_counts=seeds.sum(axis=0).tolist(),planned_stance_counts=stance.sum(axis=0).tolist(),
                precision_export=dict(source_frames=n_sparse,source_fps=30,subdivision=subdivision,
                      frames=n,fps=fps,endpoint_duration_s=(n-1)/fps,
                      reason='30Hz linear-DOF interpolation bows planted feet between knots; solve IK at finer keys',
                      object_representation='Linear position and SLERP rotation resampling, original knots exact',
                      policy_compatibility='NOT asserted: per-frame root delta channels and finite-difference derivatives are FPS sensitive; review at Priority5'),
                before=before,after=after,dense_assessment=dense,leg_target_residual_m=residuals,
                leg_target_orientation_error_deg=orientation_residuals,
                hand_preservation=hands,solver_failures=failures,solver_nfev=nfev,
                solver_restarts=restarts,additional_posture_bounds={'left_knee_joint':.02,'right_knee_joint':.02},
                knee_minimum_rad={side:float(m['dof_pos'][:,names.index(side+'_knee_joint')].min()) for side in ['left','right']},
                joint_checks=[dict(name=name,max_speed_rad_s=float(speeds[:,i].max()),
                            speed_limit_rad_s=float(limits[i]),speed_violations=int((speeds[:,i]>limits[i]+1e-6).sum()),
                            position_violations=int(((m['dof_pos'][:,i]<lower[i]-1e-6)|(m['dof_pos'][:,i]>higher[i]+1e-6)).sum())) for i,name in enumerate(names)],
                pelvis_constant_z_shift_m=-.075,pelvis_velocity_delta_max_m_s=float(abs(np.diff(m['root_pos']-baseline['root_pos'],axis=0)*fps).max()),
                pelvis_acceleration_delta_max_m_s2=float(abs(np.diff(m['root_pos']-baseline['root_pos'],n=2,axis=0)*fps**2).max()),
                input_sha256=hashes,script_sha256=sha(Path(__file__).resolve()),
                outputs={str(p):sha(p) for p in [motion,motion.with_suffix('.npz')]},
                elapsed_s=time.monotonic()-start,physical_success_verified=False,policy_run=False,training_run=False,
                limitations=['Inferred reconstructed support phases; not measured contact annotations.',
                             'No force, balance, torque, self/robot-object collision or policy simulation.',
                             'No-loop boundary/terminal settling not modified; priorities2-6 deferred.',
                             'Dense subframe checks are finite sampling, not a continuous-time guarantee.'])
    (out/'measurements.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'solver_trace.json').write_text(json.dumps(traces,indent=2,allow_nan=False)+'\n')
    def resample_source(x):
        return np.stack([np.interp(tt,np.arange(n_sparse),x[:,i]) for i in range(x.shape[1])],axis=1)
    np.savez_compressed(out/'diagnostics.npz',stance=stance,source_stance_seed=seeds,
                source_frame_coordinate=tt,source_marker_pos=resample_source(markers.reshape(n_sparse,-1)).reshape(n,2,3),
                source_speed=resample_source(np.stack([p[1] for p in phases],axis=1)),
                source_height=resample_source(np.stack([p[2] for p in phases],axis=1)),target_foot_pos=np.stack(positions,axis=1),
                before_world_body_pos=world0,after_world_body_pos=world1,
                before_capsule_gap=np.stack([v['capsule_surrogate'] for v in evaluate(baseline,urdf,names,stance)[3].values()],axis=1),
                after_capsule_gap=np.stack([clearance[f]['capsule_surrogate'] for f in FEET],axis=1))
    print(json.dumps(dict(status=report['status'],gates=gates,residuals=residuals,
                          dense=dense,elapsed_s=report['elapsed_s']),indent=2))


if __name__=='__main__':main()
