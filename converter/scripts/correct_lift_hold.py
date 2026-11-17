"""Source-informed kinematic foot correction, not dynamics/contact recovery.

Late source feet slide without measured support labels. A declared alternating
handoff is a generated correction, not evidence of actual source stepping.
"""
import argparse
import json
import pickle
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import install,ROOT,REPO,DATA,sha,write_json
install()
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation,Slerp
sys.path.insert(0,str(REPO/'legged_gym/legged_gym/scripts'))
from convert_gmr_qpos import forward_kinematics,foot_clearance,origin,FEET


def stats(a):
    a=np.asarray(a)
    return None if not a.size else dict(min=float(a.min()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()))


class Chain:
    def __init__(self,urdf,names,tip):
        by_child={j.find('child').get('link'):j for j in urdf.findall('joint')}
        joints=[]
        while tip!='pelvis':
            j=by_child[tip];joints.append(j);tip=j.find('parent').get('link')
        self.steps=[];self.indices=[]
        for j in joints[::-1]:
            p,r=origin(j);i=names.index(j.get('name')) if j.get('type')!='fixed' else None
            axis=np.fromstring(j.find('axis').get('xyz'),sep=' ') if i is not None else None
            self.steps.append((p,r,i,axis))
            if i is not None:self.indices.append(i)

    def fk(self,q,rp,rr):
        p=rp.copy();r=rr.copy()
        for v,rot,i,axis in self.steps:
            p+=r@v;r=r@rot
            if i is not None:r=r@Rotation.from_rotvec(axis*q[i]).as_matrix()
        return p,r


def sole_offset(urdf,foot,r):
    lows=[]
    for c in urdf.find("link[@name='%s']"%foot).findall('collision'):
        p,cr=origin(c);g=list(c.find('geometry'))[0]
        radius=float(g.get('radius'));half=float(g.get('length'))/2
        z=abs((r@cr)[2,2]);center=(r@p)[2]
        lows.extend([center-half*z-radius,center-half*z-radius*np.sqrt(max(0,1-z*z))])
    return min(lows)


def quintic(u):
    u=np.clip(u,0,1);return u*u*u*(10+u*(-15+6*u))


def solve(chain,q0,previous,rp,rr,tp,tr,lower,upper,indices,velocity=None,fps=30,max_nfev=100):
    lo=lower[indices]+1e-8;hi=upper[indices]-1e-8
    if velocity is not None:
        lo=np.maximum(lo,previous-velocity[indices]/fps*.999)
        hi=np.minimum(hi,previous+velocity[indices]/fps*.999)
    def f(x):
        q=q0.copy();q[indices]=x;p,r=chain.fk(q,rp,rr)
        return np.r_[(p-tp)*1000,Rotation.from_matrix(tr.T@r).as_rotvec()*100,
                     (x-q0[indices])*.01,(x-previous)*.01]
    initial=np.minimum(np.maximum(previous,lo+1e-10),hi-1e-10)
    res=least_squares(f,initial,bounds=(lo,hi),max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-7)
    q=q0.copy();q[indices]=res.x;p,r=chain.fk(q,rp,rr)
    return res.x,float(np.linalg.norm(p-tp)),float(Rotation.from_matrix(tr.T@r).magnitude()),res.nfev


def targets(m,urdf,names,human,swing_height=.04,vertical_fraction=.22):
    n=len(m['root_pos']);fps=float(m['fps']);lp,lr=forward_kinematics(urdf,m['dof_pos'],names,m['link_body_list'])
    rr=Rotation.from_quat(m['root_rot']).as_matrix();world=m['root_pos'][:,None]+np.einsum('tij,tbj->tbi',rr,lp)
    # Ten-frame net XY movement is resistant to the source's alternating-step
    # amplitudes and never interprets one high-speed interval as a new step.
    markers=human[:,[10,11]];speed=[]
    for side in range(2):
        x=markers[:,side];out=[]
        for k in range(n):
            a=max(0,k-5);b=min(n-1,k+5)
            out.append(np.linalg.norm(x[b,:2]-x[a,:2])*fps/(b-a))
        speed.append(np.asarray(out))
    active=np.maximum(*speed)>.02
    # The large initial few-frame settling transient is not treated as gait.
    active[:max(3,int(.5*fps))]=False
    idx=np.flatnonzero(active)
    start=max(1,int(idx[0])-2) if len(idx) else n-2
    end=min(n-3,int(idx[-1])+2) if len(idx) else n-2
    first=int(speed[1][start:min(end,start+12)].mean()>speed[0][start:min(end,start+12)].mean())
    swings=[[],[]]
    if end-start>=int(2*fps):
        boundaries=np.round(np.linspace(start,end,5)).astype(int)
        for k in range(4):swings[(first+k)%2].append((int(boundaries[k]+(1 if k else 0)),int(boundaries[k+1])))
    support=np.ones((n,2),bool)
    for side in range(2):
        for a,b in swings[side]:support[a+1:b,side]=False
    tp=[];tr=[];anchor_details=[]
    for side,foot in enumerate(FEET):
        fi=m['link_body_list'].index(foot);wp=gaussian_filter1d(world[:,fi],2,axis=0,mode='nearest')
        wr=rr@lr[foot];yaw=np.unwrap(np.arctan2(wr[:,1,0],wr[:,0,0]))
        p=np.zeros((n,3));y=np.zeros(n);segs=[]
        edges=[0]+[v for a,b in swings[side] for v in (a+1,b)]+[n]
        for a,b in zip(edges[::2],edges[1::2]):
            if a==0:
                chosen=np.arange(min(10,max(1,start//3)),max(11,start-6))
            else:
                chosen=np.arange(a,min(b,a+6))
            pp=np.median(wp[chosen],axis=0);yy=float(np.median(yaw[chosen]));r=Rotation.from_rotvec([0,0,yy]).as_matrix()
            pp[2]=.0005-sole_offset(urdf,foot,r)
            p[a:b]=pp;y[a:b]=yy;segs.append(dict(start=a,stop=b,source_frames=chosen.tolist(),anchor=pp.tolist(),yaw=yy))
        for a,b in swings[side]:
            fraction=vertical_fraction
            u=np.arange(b-a+1)/(b-a);w=quintic((u-fraction)/(1-2*fraction))
            p[a:b+1]=(1-w[:,None])*p[a]+w[:,None]*p[b]
            y[a:b+1]=(1-w)*y[a]+w*y[b]
            lift=np.where(u<fraction,quintic(u/fraction),np.where(u>1-fraction,quintic((1-u)/fraction),1.))
            p[a:b+1,2]+=swing_height*lift
        tp.append(p);tr.append(Rotation.from_rotvec(np.c_[np.zeros(n),np.zeros(n),y]).as_matrix());anchor_details.append(segs)
    report=dict(kind='GENERATED_KINEMATIC_CONTACT_CORRECTION_NOT_RECOVERED_PHYSICAL_CONTACT',
        source_window_frames=10,source_motion_threshold_m_s=.02,source_activity_start_end=[start,end],
        generated_swing_intervals_open=swings,anchors=anchor_details,swing_height_m=swing_height,vertical_fraction=vertical_fraction,
        minimum_support_feet_per_sample=int(support.sum(axis=1).min()),
        source_marker_speed_window_m_s=[stats(s) for s in speed],
        source_late_contact_labels_known=False,initial_transient_policy='replace with median static support, not an inferred swing')
    return tp,tr,report,(lp,lr,rr,world)


def dense_assess(m,urdf,names):
    n=len(m['root_pos']);fps=m['fps'];sub=20
    t=np.arange((n-1)*sub+1)/sub;lo=np.minimum(t.astype(int),n-2);w=(t-lo)[:,None]
    q=(1-w)*m['dof_pos'][lo]+w*m['dof_pos'][lo+1]
    rp=(1-w)*m['root_pos'][lo]+w*m['root_pos'][lo+1]
    rq=Slerp(np.arange(n),Rotation.from_quat(m['root_rot']))(t).as_quat();rr=Rotation.from_quat(rq).as_matrix()
    lp,lr=forward_kinematics(urdf,q,names,m['link_body_list']);world=rp[:,None]+np.einsum('tij,tbj->tbi',rr,lp)
    clear=foot_clearance(urdf,lp,lr,m['link_body_list'],rp,rr);support=[];speeds=[];gaps={}
    dt=1/(fps*sub)
    for foot in FEET:
        c=clear[foot];gaps[foot]={key:stats(v) for key,v in c.items()}
        support.append(np.maximum(abs(c['cylinder']),abs(c['capsule_surrogate']))<=.005)
        pos=world[:,m['link_body_list'].index(foot)];rot=rr@lr[foot]
        for collision in urdf.find("link[@name='%s']"%foot).findall('collision'):
            cp,cr=origin(collision);geom=list(collision.find('geometry'))[0];radius=float(geom.get('radius'));half=float(geom.get('length'))/2
            for sign in [-1,0,1]:
                point=pos+np.einsum('tij,j->ti',rot,cp+sign*half*cr[:,2]);gap=point[:,2]-radius
                near=(gap[:-1]<.005)|(gap[1:]<.005);v=np.linalg.norm(np.diff(point[:,:2],axis=0),axis=1)/dt
                speeds.extend(v[near])
    return dict(samples=len(t),sample_rate_hz=fps*sub,min_gap_m=min(v.min() for c in clear.values() for v in c.values()),
                unsupported_samples=int((~np.stack(support,axis=1).any(axis=1)).sum()),foot_gaps=gaps,near_floor_probe_speed_m_s=stats(speeds),
                note='Numerical dense FK assessment; actual loader and self collision are independently validated')


def precision_refine(args):
    """Reproject only legs at precision knots; upper continuous paths retained."""
    source=args.precision_from.resolve(strict=True);out=DATA/'candidates'/args.attempt
    if DATA not in source.parents or out.exists():raise ValueError('Source in DATA and new candidate directory required')
    out.mkdir(parents=True)
    m=pickle.load(source.open('rb'));n=len(m['root_pos']);sub=args.subdivision
    t=np.arange((n-1)*sub+1,dtype=float)/sub;lo=np.minimum(t.astype(int),n-2);w=(t-lo)[:,None]
    new={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in m.items()}
    new['root_pos']=(1-w)*m['root_pos'][lo]+w*m['root_pos'][lo+1]
    new['dof_pos']=(1-w)*m['dof_pos'][lo]+w*m['dof_pos'][lo+1]
    new['root_rot']=Slerp(np.arange(n),Rotation.from_quat(m['root_rot']))(t).as_quat()
    new['root_rot'][::sub]=m['root_rot'];new['fps']=m['fps']*sub
    urdf=ET.parse(REPO/'assets/g1/g1_custom_collision_29dof.urdf').getroot();joints=[j for j in urdf.findall('joint') if j.get('type')!='fixed'];names=[j.get('name') for j in joints]
    lower=np.array([float(j.find('limit').get('lower')) for j in joints]);upper=np.array([float(j.find('limit').get('upper')) for j in joints]);vmax=np.array([float(j.find('limit').get('velocity')) for j in joints])
    with np.load(args.foot_targets.resolve(strict=True)) as z:pp=z['positions'];rot=z['rotations']
    if len(pp)!=n:raise ValueError('Foot target knots differ from source')
    rr=Rotation.from_quat(new['root_rot']).as_matrix();traces={};started=time.monotonic()
    for side,foot in enumerate(FEET):
        tp=(1-w)*pp[lo,side]+w*pp[lo+1,side]
        tr=Slerp(np.arange(n),Rotation.from_matrix(rot[:,side]))(t).as_matrix()
        chain=Chain(urdf,names,foot);idx=np.array(chain.indices);previous=new['dof_pos'][0,idx].copy();trace=[]
        for k in range(len(t)):
            reference=new['dof_pos'][k].copy()
            values,pe,re,calls=solve(chain,reference,previous,new['root_pos'][k],rr[k],tp[k],tr[k],lower,upper,idx,
                                    velocity=vmax if k else None,fps=new['fps'],max_nfev=100)
            new['dof_pos'][k,idx]=values;previous=values;trace.append([pe,re,calls])
        traces[foot]=dict(position_m=stats(np.asarray(trace)[:,0]),rotation_rad=stats(np.asarray(trace)[:,1]))
        print('PRECISION_LEG',foot,traces[foot],flush=True)
    lp,new_lr=forward_kinematics(urdf,new['dof_pos'],names,new['link_body_list']);new['local_body_pos']=lp
    original=pickle.load((DATA/'baseline/lift_hold_001_g1.pkl').open('rb'))
    oq=(1-w)*original['dof_pos'][lo]+w*original['dof_pos'][lo+1]
    orp=(1-w)*original['root_pos'][lo]+w*original['root_pos'][lo+1]
    olp,olr=forward_kinematics(urdf,oq,names,original['link_body_list'])
    old_world=orp[:,None]+np.einsum('tij,tbj->tbi',rr,olp)
    new_world=new['root_pos'][:,None]+np.einsum('tij,tbj->tbi',rr,lp)
    hand_changes={}
    for hand in ['left_rubber_hand','right_rubber_hand']:
        j=new['link_body_list'].index(hand)
        hand_changes[hand]=dict(origin_error_m=stats(np.linalg.norm(new_world[:,j]-old_world[:,j],axis=1)),
            rotation_error_deg=stats(np.degrees(Rotation.from_matrix(np.swapaxes(olr[hand],1,2)@new_lr[hand]).magnitude())))
    cp=out/'lift_hold_001_g1.pkl'
    with cp.open('xb') as f:pickle.dump(new,f,protocol=4)
    with np.load(source.with_suffix('.npz'),allow_pickle=False) as z:
        object_data={}
        for key in z.files:
            value=z[key]
            if key=='rot':
                object_data[key]=Slerp(np.arange(n),Rotation.from_quat(value))(t).as_quat();object_data[key][::sub]=value
            elif value.ndim and len(value)==n:
                shape=(len(w),)+(1,)*(value.ndim-1);weight=(t-lo).reshape(shape)
                object_data[key]=(1-weight)*value[lo]+weight*value[lo+1]
            else:object_data[key]=value
    np.savez_compressed(cp.with_suffix('.npz'),**object_data)
    after=dense_assess(new,urdf,names);q=new['dof_pos'];speed=abs(np.diff(q,axis=0))*new['fps']
    report=dict(status='PENDING_INDEPENDENT_VALIDATION',attempt=args.attempt,kind='PRECISION_LEG_REPROJECTION',
        source=str(source),source_sha256=sha(source),target_sha256=sha(args.foot_targets),frames=len(t),fps=new['fps'],
        duration_s=(len(t)-1)/new['fps'],source_duration_s=(n-1)/m['fps'],
        after=after,leg_target_errors=traces,
        additional_hand_correction_vs_original_baseline=hand_changes,
        source_leg_native_knots_change_max_rad=float(abs(new['dof_pos'][::sub,:12]-m['dof_pos'][:,:12]).max()),
        source_leg_max_speed_rad_s=float((abs(np.diff(m['dof_pos'][:,:12],axis=0))*m['fps']).max()),
        corrected_leg_max_speed_rad_s=float(speed[:,:12].max()),
        root_z_correction_vs_original_baseline_m=stats(new['root_pos'][:,2]-orp[:,2]),
        position_limit_violations=int(((q<lower-1e-6)|(q>upper+1e-6)).sum()),speed_limit_violations=int((speed>vmax+1e-6).sum()),
        root_native_knots_exact=bool(np.array_equal(new['root_pos'][::sub],m['root_pos']) and np.array_equal(new['root_rot'][::sub],m['root_rot'])),
        upper_native_knots_exact=bool(np.array_equal(new['dof_pos'][::sub,12:],m['dof_pos'][:,12:])),
        upper_continuous_path='Exactly the input 30Hz linear-joint/SLERP-root path; only legs reprojected',
        object_path='Same linear position/SLERP quaternion path and source clock, native knots retained',
        elapsed_solver_s=time.monotonic()-started,quality_gates_sha256=sha(ROOT/'config/quality_gates.json'),
        self_collision_validated=False,physical_success_verified=False,policy_or_training_run=False)
    report['output_sha256']={str(f):sha(f) for f in [cp,cp.with_suffix('.npz')]}
    write_json(ROOT/'reports'/f'foot_{args.attempt}.json',report)
    print(json.dumps({k:report[k] for k in ['attempt','frames','fps','after','position_limit_violations','speed_limit_violations','elapsed_solver_s']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--attempt',default='attempt01');parser.add_argument('--drop',type=float);parser.add_argument('--no-velocity-boxes',action='store_true')
    parser.add_argument('--swing-height',type=float,default=.04);parser.add_argument('--vertical-fraction',type=float,default=.22)
    parser.add_argument('--precision-from',type=Path);parser.add_argument('--foot-targets',type=Path,default=DATA/'candidates/attempt02/foot_targets.npz');parser.add_argument('--subdivision',type=int,default=4)
    args=parser.parse_args();out=DATA/'candidates'/args.attempt
    if args.precision_from:
        if args.subdivision not in [2,4,8]:raise ValueError('Unreviewed precision subdivision')
        return precision_refine(args)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    p=DATA/'baseline/lift_hold_001_g1.pkl';m=pickle.load(p.open('rb'));n=len(m['root_pos']);fps=m['fps']
    urdf=ET.parse(REPO/'assets/g1/g1_custom_collision_29dof.urdf').getroot();joints=[j for j in urdf.findall('joint') if j.get('type')!='fixed'];names=[j.get('name') for j in joints]
    lower=np.array([float(j.find('limit').get('lower')) for j in joints]);upper=np.array([float(j.find('limit').get('upper')) for j in joints]);vmax=np.array([float(j.find('limit').get('velocity')) for j in joints])
    with np.load(DATA/'aligned/lift_hold_001_aligned.npz') as z:human=z['human_joint_pos']
    if not .025<=args.swing_height<=.08 or not .1<=args.vertical_fraction<=.3:raise ValueError('Unsupported swing geometry')
    tp,tr,schedule,state=targets(m,urdf,names,human,args.swing_height,args.vertical_fraction);lp,lr,rr,world=state
    before=dense_assess(m,urdf,names);chains=[Chain(urdf,names,f) for f in FEET];indices=[np.array(c.indices) for c in chains]
    search=[]
    measured=np.median([before['foot_gaps'][f]['capsule_surrogate']['median'] for f in FEET])
    drops=[args.drop] if args.drop is not None else np.round(np.arange(max(.0,measured-.035),measured+.071,.01),4)
    for drop in drops:
        worst=0.;rotworst=0.
        for side in range(2):
            for k in np.unique(np.linspace(0,n-1,9).astype(int)):
                q=m['dof_pos'][k].copy();idx=indices[side];seed=q[idx].copy()
                if seed[3]<.15:seed[3]=.3
                result,err,re,_=solve(chains[side],q,seed,m['root_pos'][k]-[0,0,drop],rr[k],tp[side][k],tr[side][k],lower,upper,idx,max_nfev=200)
                worst=max(worst,err);rotworst=max(rotworst,re)
        search.append(dict(drop_m=float(drop),max_position_error_m=worst,max_rotation_error_rad=rotworst))
        print('DROP',search[-1],flush=True)
        if worst<.0005 and rotworst<.005:break
    feasible=[s for s in search if s['max_position_error_m']<.0005 and s['max_rotation_error_rad']<.005]
    chosen=min(feasible,key=lambda s:s['drop_m']) if feasible else min(search,key=lambda s:s['max_position_error_m'])
    drop=chosen['drop_m'];candidate={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in m.items()};candidate['root_pos'][:,2]-=drop
    errors={};started=time.monotonic()
    for side,chain in enumerate(chains):
        idx=indices[side];previous=m['dof_pos'][0,idx].copy();trace=[]
        for k in range(n):
            values,pe,re,calls=solve(chain,m['dof_pos'][k],previous,candidate['root_pos'][k],rr[k],tp[side][k],tr[side][k],lower,upper,idx,
                velocity=vmax if k and not args.no_velocity_boxes else None,fps=fps,max_nfev=150)
            candidate['dof_pos'][k,idx]=values;previous=values;trace.append([pe,re,calls])
        errors[FEET[side]]=dict(position_m=stats(np.asarray(trace)[:,0]),rotation_rad=stats(np.asarray(trace)[:,1]))
        print('LEG',FEET[side],errors[FEET[side]],flush=True)
    hand_metrics={}
    for side,hand in enumerate(['left_rubber_hand','right_rubber_hand']):
        chain=Chain(urdf,names,hand);idx=np.array(chain.indices[-7:]);previous=m['dof_pos'][0,idx].copy();trace=[]
        target_pos=world[:,m['link_body_list'].index(hand)];target_rot=rr@lr[hand]
        for k in range(n):
            values,pe,re,calls=solve(chain,candidate['dof_pos'][k],previous,candidate['root_pos'][k],rr[k],target_pos[k],target_rot[k],lower,upper,idx,
                velocity=vmax if k and not args.no_velocity_boxes else None,fps=fps,max_nfev=150)
            candidate['dof_pos'][k,idx]=values;previous=values;trace.append([pe,re,calls])
        hand_metrics[hand]=dict(position_m=stats(np.asarray(trace)[:,0]),rotation_rad=stats(np.asarray(trace)[:,1]))
        print('HAND',hand,hand_metrics[hand],flush=True)
    newlp,_=forward_kinematics(urdf,candidate['dof_pos'],names,candidate['link_body_list']);candidate['local_body_pos']=newlp
    after=dense_assess(candidate,urdf,names);q=candidate['dof_pos'];speed=abs(np.diff(q,axis=0))*fps
    report=dict(status='PENDING_INDEPENDENT_VALIDATION',attempt=args.attempt,root_z_constant_correction_m=-drop,
        before=before,after=after,schedule=schedule,drop_search=search,leg_target_errors=errors,hand_world_preservation_errors=hand_metrics,
        position_limit_violations=int(((q<lower-1e-6)|(q>upper+1e-6)).sum()),speed_limit_violations=int((speed>vmax+1e-6).sum()),
        maximum_joint_speed_rad_s=float(speed.max()),velocity_boxes_used=not args.no_velocity_boxes,
        elapsed_solver_s=time.monotonic()-started,source_sha256=sha(p),quality_gates_sha256=sha(ROOT/'config/quality_gates.json'),
        root_xy_preserved=bool(np.array_equal(candidate['root_pos'][:,:2],m['root_pos'][:,:2])),root_orientation_preserved=bool(np.array_equal(candidate['root_rot'],m['root_rot'])),
        physical_success_verified=False,self_collision_validated=False,policy_or_training_run=False)
    cp=out/'lift_hold_001_g1.pkl'
    with cp.open('xb') as f:pickle.dump(candidate,f,protocol=4)
    object_data=np.load(p.with_suffix('.npz'),allow_pickle=False)
    np.savez_compressed(cp.with_suffix('.npz'),**{k:object_data[k] for k in object_data.files})
    np.savez_compressed(out/'foot_targets.npz',positions=np.stack(tp,axis=1),rotations=np.stack(tr,axis=1))
    report['output_sha256']={str(f):sha(f) for f in [cp,cp.with_suffix('.npz')]}
    write_json(ROOT/'reports'/f'foot_{args.attempt}.json',report)
    print(json.dumps({k:report[k] for k in ['attempt','root_z_constant_correction_m','position_limit_violations','speed_limit_violations','after','elapsed_solver_s']}),flush=True)


if __name__=='__main__':main()
