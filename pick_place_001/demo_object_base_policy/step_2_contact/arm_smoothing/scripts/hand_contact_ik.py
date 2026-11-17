"""Analytic arm contact Jacobians and sparse point/normal trajectory fitting.

Only joint position bounds are hard constraints in the least-squares solver.
Velocity hinges are optimization penalties: callers MUST inspect returned
strict gates and independently validate geometry/loader interpolation.
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, diags, eye, kron, vstack
from scipy.spatial.transform import Rotation


def _axis_rotation(angles, axis):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    c, s = np.cos(angles), np.sin(angles)
    return (c[:, None, None] * np.eye(3)
            + (1-c)[:, None, None] * np.outer(axis, axis)
            + s[:, None, None] * skew)


class ArmContactKinematics:
    """Vectorized URDF-chain contact position/normal and analytic derivatives.

    ``chain`` is compatible with correct_foot_contact.Chain. All joints outside
    ``indices`` remain exactly at their per-frame ``full_reference`` values.
    """
    def __init__(self, chain, full_reference, root_pos, root_rot,
                 pad_local, normal_local, indices=None):
        self.chain = chain
        self.full_reference = np.atleast_2d(np.asarray(full_reference, float))
        self.root_pos = np.atleast_2d(np.asarray(root_pos, float))
        self.root_rot = np.asarray(root_rot, float)
        if self.root_rot.ndim == 2:
            self.root_rot = self.root_rot[None]
        self.pad = np.asarray(pad_local, float)
        self.normal = np.asarray(normal_local, float)
        self.normal /= np.linalg.norm(self.normal)
        self.indices = np.asarray(chain.indices[-7:] if indices is None else indices, int)
        self.active = {int(j): k for k, j in enumerate(self.indices)}
        self.n = len(self.full_reference)
        self.d = len(self.indices)
        if self.root_pos.shape != (self.n, 3) or self.root_rot.shape != (self.n, 3, 3):
            raise ValueError('Per-frame root and reference shapes differ')
        if self.d < 1:
            raise ValueError('Expected active joint indices')

    def evaluate(self, arm_angles, jacobian=True):
        q = np.asarray(arm_angles, float).reshape(self.n, self.d)
        p, r = self.root_pos.copy(), self.root_rot.copy()
        if jacobian:
            # A shared waist+two-arm index set contains joints not in this
            # particular chain. Their analytic columns are exactly zero.
            axes = np.zeros((self.n, self.d, 3))
            origins = np.zeros((self.n, self.d, 3))
        for xyz, rotation, index, axis in self.chain.steps:
            p += np.einsum('nij,j->ni', r, xyz)
            r = r @ rotation
            if index is None:
                continue
            if index in self.active:
                j = self.active[index]
                if jacobian:
                    unit = np.asarray(axis)/np.linalg.norm(axis)
                    axes[:, j] = np.einsum('nij,j->ni', r, unit)
                    origins[:, j] = p
                angle = q[:, j]
            else:
                angle = self.full_reference[:, index]
            r = r @ _axis_rotation(angle, axis)
        point = p + np.einsum('nij,j->ni', r, self.pad)
        normal = np.einsum('nij,j->ni', r, self.normal)
        if not jacobian:
            return point, normal
        jp = np.cross(axes, point[:, None]-origins).transpose(0, 2, 1)
        jn = np.cross(axes, normal[:, None]).transpose(0, 2, 1)
        return point, normal, jp, jn


class CoupledContactKinematics:
    """Two or more chains with a shared waist/arm variable vector.

    evaluate returns point/normal [N,K,3], derivatives [N,K,3,D].
    task returns weighted residual [N,6*K] and Jacobian [N,6*K,D], ordered
    as all contact XYZ points followed by all contact XYZ normals.
    """
    def __init__(self, chains, full_reference, root_pos, root_rot,
                 pads_local, normals_local, indices=None):
        if indices is None:
            indices=list(dict.fromkeys(j for chain in chains for j in chain.indices))
        self.models=[ArmContactKinematics(c,full_reference,root_pos,root_rot,p,nn,indices)
                     for c,p,nn in zip(chains,pads_local,normals_local)]
        if not self.models:raise ValueError('At least one contact chain is required')
        self.n,self.d=self.models[0].n,self.models[0].d
        self.indices=self.models[0].indices
        self.full_reference=self.models[0].full_reference
        self.k=len(self.models)

    def evaluate(self, angles, jacobian=True):
        values=[m.evaluate(angles,jacobian) for m in self.models]
        return tuple(np.stack([v[i] for v in values],axis=1) for i in range(4 if jacobian else 2))

    def task(self, angles, target_points, target_normals,
             position_weight=3000.,normal_weight=500.):
        p,nn,jp,jn=self.evaluate(angles)
        residual=np.concatenate([(p-np.asarray(target_points)).reshape(self.n,-1)*position_weight,
                                 (nn-np.asarray(target_normals)).reshape(self.n,-1)*normal_weight],axis=1)
        jac=np.concatenate([jp.reshape(self.n,-1,self.d)*position_weight,
                            jn.reshape(self.n,-1,self.d)*normal_weight],axis=1)
        return residual,jac


def contact_state(chain, full_angles, root_pos, root_rot, pad_local,
                  normal_local, indices=None):
    """Single-frame ``(contact, normal, J_contact[3,7], J_normal[3,7])``."""
    kin = ArmContactKinematics(chain, full_angles, root_pos, root_rot,
                               pad_local, normal_local, indices)
    result = kin.evaluate(kin.full_reference[:, kin.indices])
    return tuple(v[0] for v in result)


def solve_contact_pose(chain, full_angles, root_pos, root_rot, pad_local,
                       normal_local, target_point, target_normal, lower, upper,
                       previous=None, seeds=None, indices=None,
                       position_weight=3000., normal_weight=500.,
                       reference_weight=.1, previous_weight=.02, max_nfev=120):
    """Analytic per-frame IK; candidates ranked feasible first, then continuity.

    ``lower``/``upper`` may be full-robot or seven-element arm bounds. Returned
    optimizer success is separate from the explicit geometric feasibility.
    """
    kin = ArmContactKinematics(chain, full_angles, root_pos, root_rot,
                               pad_local, normal_local, indices)
    d=kin.d
    reference = kin.full_reference[0, kin.indices]
    previous = reference if previous is None else np.asarray(previous, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if lower.shape != (d,): lower = lower[kin.indices]
    if upper.shape != (d,): upper = upper[kin.indices]
    target_point = np.asarray(target_point, float)
    target_normal = np.asarray(target_normal, float)
    target_normal /= np.linalg.norm(target_normal)
    cache = {}
    def calculate(x):
        if 'x' not in cache or not np.array_equal(x, cache['x']):
            p, n, jp, jn = kin.evaluate(x[None])
            cache.update(x=x.copy(), point=p[0], normal=n[0], jp=jp[0], jn=jn[0])
        return cache
    def fun(x):
        v=calculate(x)
        return np.r_[(v['point']-target_point)*position_weight,
                      (v['normal']-target_normal)*normal_weight,
                      (x-reference)*reference_weight,
                      (x-previous)*previous_weight]
    def jac(x):
        v=calculate(x)
        return np.vstack([v['jp']*position_weight, v['jn']*normal_weight,
                          np.eye(d)*reference_weight, np.eye(d)*previous_weight])
    candidates=[]
    for seed_index, seed in enumerate([previous,reference] if seeds is None else seeds):
        result=least_squares(fun,np.clip(seed,lower+1e-10,upper-1e-10),jac=jac,
                             bounds=(lower,upper),max_nfev=max_nfev,
                             ftol=1e-9,xtol=1e-9,gtol=1e-8)
        p,n=kin.evaluate(result.x[None],False)
        pe=float(np.linalg.norm(p[0]-target_point))
        ne=float(np.arccos(np.clip(np.dot(n[0],target_normal),-1,1)))
        feasible=pe<=.005 and ne<=np.radians(5)
        continuity=float(np.linalg.norm(result.x-previous))
        posture=float(np.linalg.norm(result.x-reference))
        # Contact-invalid candidates never win solely for a small q change.
        rank=(not feasible, pe*1000+ne*100 if not feasible else 0.,
              continuity+.1*posture)
        candidates.append((rank,result,pe,ne,seed_index))
    _,result,pe,ne,seed_index=min(candidates,key=lambda v:v[0])
    return dict(q=result.x,position_error_m=pe,normal_error_rad=ne,
                contact_feasible=pe<=.005 and ne<=np.radians(5),
                optimizer_success=bool(result.success),seed_index=seed_index,
                nfev=int(result.nfev),message=str(result.message))


def trajectory_metrics(kin, q, target_points, target_normals, lower, upper,
                       speed_limits, fps):
    p,n=kin.evaluate(q,False)
    pe=np.linalg.norm(p-target_points,axis=1)
    ne=np.degrees(np.arccos(np.clip(np.sum(n*target_normals,axis=1),-1,1)))
    velocity=np.abs(np.diff(q,axis=0))*fps
    violation=np.maximum(velocity-speed_limits,0.)
    gates=dict(position_5mm=bool(pe.max()<=.005),normal_5deg=bool(ne.max()<=5),
               joint_positions=bool(((q>=lower-1e-9)&(q<=upper+1e-9)).all()),
               joint_speeds=bool((violation<=1e-6).all()),finite=bool(np.isfinite(q).all()))
    return dict(gates=gates,all_gates=all(gates.values()),
                position_max_m=float(pe.max()),position_p95_m=float(np.percentile(pe,95)),
                normal_max_deg=float(ne.max()),normal_p95_deg=float(np.percentile(ne,95)),
                maximum_speed_rad_s=velocity.max(axis=0).tolist(),
                speed_violation_count=int((violation>1e-6).sum()),
                max_speed_excess_rad_s=float(violation.max()),
                max_speed_ratio=float((velocity/speed_limits).max()))


def smooth_contact_trajectory(chain, full_reference, root_pos, root_rot,
                              pad_local, normal_local, target_points,
                              target_normals, initial, lower, upper, speed_limits,
                              fps=120., indices=None, reference_weight=.05,
                              velocity_smooth_weight=.003, acceleration_weight=.00005,
                              position_weight=3000., normal_weight=500.,
                              hinge_weights=(10.,1000.,100000.), max_nfev=100,
                              progress=None):
    """Sparse global re-projection with velocity penalties and strict diagnostics.

    Returns q, per-stage optimizer evidence, and strict final metrics. A failed
    velocity gate remains a failure; no output clipping/smoothing is concealed.
    """
    kin=ArmContactKinematics(chain,full_reference,root_pos,root_rot,pad_local,normal_local,indices)
    n,d=kin.n,kin.d
    lower,upper,speed_limits=map(lambda x:np.asarray(x,float),(lower,upper,speed_limits))
    if lower.shape!=(d,):lower=lower[kin.indices]
    if upper.shape!=(d,):upper=upper[kin.indices]
    if speed_limits.shape!=(d,):speed_limits=speed_limits[kin.indices]
    targets=np.asarray(target_points,float)
    normals=np.asarray(target_normals,float)
    normals=normals/np.linalg.norm(normals,axis=1,keepdims=True)
    reference=kin.full_reference[:,kin.indices]
    q=np.asarray(initial,float).reshape(n,d).copy()
    if ((q<lower)|(q>upper)).any():
        raise ValueError('Initial q outside supplied hard position bounds')
    identity=eye(n*d,format='csr')
    d1=kron(diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n)),eye(d),format='csr')*fps
    d2=kron(diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n)),eye(d),format='csr')*fps**2
    # CSR indices for independent 6x7 per-frame contact blocks.
    row=np.repeat(np.arange(n*6),d)
    col=(np.repeat(np.arange(n)*d,6*d)+np.tile(np.arange(d),n*6))
    stages=[]
    start=time.monotonic()
    initial_metrics=trajectory_metrics(kin,q,targets,normals,lower,upper,speed_limits,fps)
    for hinge in hinge_weights:
        cache={}
        def calc(x):
            if 'x' in cache and np.array_equal(x,cache['x']):return cache
            angles=x.reshape(n,d)
            p,nn,jp,jn=kin.evaluate(angles)
            contact=np.concatenate([(p-targets)*position_weight,(nn-normals)*normal_weight],axis=1)
            blocks=np.concatenate([jp*position_weight,jn*normal_weight],axis=1)
            sparse_contact=csr_matrix((blocks.ravel(),(row,col)),shape=(n*6,n*d))
            velocity=d1@x
            excess=np.abs(velocity)-np.tile(speed_limits,n-1)
            active=excess>0
            fun=np.r_[contact.ravel(),(x-reference.ravel())*reference_weight,
                      velocity*velocity_smooth_weight,(d2@x)*acceleration_weight,
                      np.maximum(excess,0)*hinge]
            jac=vstack([sparse_contact,identity*reference_weight,d1*velocity_smooth_weight,
                        d2*acceleration_weight,diags(np.sign(velocity)*active*hinge)@d1],format='csr')
            cache.update(x=x.copy(),fun=fun,jac=jac)
            return cache
        if progress:progress(dict(event='stage_start',hinge_weight=hinge,elapsed_s=time.monotonic()-start))
        result=least_squares(lambda x:calc(x)['fun'],q.ravel(),jac=lambda x:calc(x)['jac'],
                    bounds=(np.tile(lower,n),np.tile(upper,n)),method='trf',tr_solver='lsmr',
                    x_scale='jac',max_nfev=max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-8,
                    tr_options=dict(atol=1e-6,btol=1e-6,maxiter=160))
        q=result.x.reshape(n,d)
        metrics=trajectory_metrics(kin,q,targets,normals,lower,upper,speed_limits,fps)
        stage=dict(hinge_weight=hinge,nfev=int(result.nfev),njev=int(result.njev),
                   optimizer_success=bool(result.success),message=str(result.message),
                   cost=float(result.cost),metrics=metrics,elapsed_s=time.monotonic()-start)
        stages.append(stage)
        if progress:progress(dict(event='stage_done',**stage))
        if metrics['all_gates']:break
    return dict(q=q,initial_metrics=initial_metrics,metrics=metrics,stages=stages,
                elapsed_s=time.monotonic()-start,
                scope='Point/normal and joint trajectory fit only; no collision, force, or interpolated-loader validation')


def _fixtures():
    import pickle
    import xml.etree.ElementTree as ET
    here=Path(__file__).resolve().parent
    sys.path.insert(0,str(here))
    from correct_foot_contact import Chain,WORK,REPO
    path=WORK/'execution/priority_1_foot_contact/attempt09/interpose_floorlamp_g1_foot_contact.pkl'
    with path.open('rb') as f:m=pickle.load(f)
    plan=json.loads((WORK/'execution/priority_2_hand_contact/attempt02/contact_plan.json').read_text())
    urdf=ET.parse(REPO/'assets/g1/g1_custom_collision_29dof.urdf').getroot()
    joints=[j for j in urdf.findall('joint') if j.get('type')!='fixed']
    names=[j.get('name') for j in joints]
    chain=Chain(urdf,names,'left_rubber_hand')
    spec=plan['hands']['left'];pad=np.array(spec['pad_local']);normal=np.array(spec['pad_normal_local'])
    rr=Rotation.from_quat(m['root_rot']).as_matrix()
    ids=np.array(chain.indices[-7:]);rng=np.random.default_rng(20261110)
    frames=rng.choice(len(rr),10,replace=False)
    numerical=[];fk=[]
    for t in frames:
        full=m['dof_pos'][t].copy();full[ids]+=rng.uniform(-.1,.1,7)
        p,n,jp,jn=contact_state(chain,full,m['root_pos'][t],rr[t],pad,normal)
        pp,pr=chain.fk(full,m['root_pos'][t],rr[t])
        fk.append(max(abs(pp+pr@pad-p).max(),abs(pr@normal-n).max()))
        numeric=np.zeros((6,7))
        for i,j in enumerate(ids):
            plus=full.copy();minus=full.copy();plus[j]+=1e-7;minus[j]-=1e-7
            pa,ra=chain.fk(plus,m['root_pos'][t],rr[t]);pb,rb=chain.fk(minus,m['root_pos'][t],rr[t])
            numeric[:,i]=np.r_[pa+ra@pad-pb-rb@pad,ra@normal-rb@normal]/2e-7
        numerical.append(float(abs(numeric-np.vstack([jp,jn])).max()))
    kin=ArmContactKinematics(chain,m['dof_pos'][frames],m['root_pos'][frames],rr[frames],pad,normal)
    points,normals,jp,jn=kin.evaluate(m['dof_pos'][frames][:,ids])
    batch_error=[]
    for i,t in enumerate(frames):
        single=contact_state(chain,m['dof_pos'][t],m['root_pos'][t],rr[t],pad,normal)
        batch_error.append(max(abs(points[i]-single[0]).max(),abs(normals[i]-single[1]).max(),
                               abs(jp[i]-single[2]).max(),abs(jn[i]-single[3]).max()))
    # A shared 17-variable waist/two-arm model must leave the other arm's
    # columns zero in each individual chain and differentiate common waist.
    chains=[chain,Chain(urdf,names,'right_rubber_hand')]
    shared=np.arange(12,29)
    coupled_error=[];unused_error=[]
    pads=[plan['hands'][s]['pad_local'] for s in ['left','right']]
    normals_local=[plan['hands'][s]['pad_normal_local'] for s in ['left','right']]
    for t in frames[:3]:
        full=m['dof_pos'][t].copy()
        model=CoupledContactKinematics(chains,full,m['root_pos'][t],rr[t],pads,normals_local,shared)
        x=full[shared]+rng.uniform(-.05,.05,len(shared))
        residual,jac=model.task(x[None],np.zeros((1,2,3)),np.zeros((1,2,3)),1.,1.)
        num=np.empty((12,len(shared)))
        for j in range(len(shared)):
            plus=x.copy();minus=x.copy();plus[j]+=1e-7;minus[j]-=1e-7
            a,_=model.task(plus[None],np.zeros((1,2,3)),np.zeros((1,2,3)),1.,1.)
            b,_=model.task(minus[None],np.zeros((1,2,3)),np.zeros((1,2,3)),1.,1.)
            num[:,j]=(a-b)[0]/2e-7
        coupled_error.append(float(abs(jac[0]-num).max()))
        _,_,jp,jn=model.evaluate(x[None])
        unused_error.append(max(float(abs(jp[0,0,:,10:]).max()),float(abs(jn[0,0,:,10:]).max()),
                                float(abs(jp[0,1,:,3:10]).max()),float(abs(jn[0,1,:,3:10]).max())))
    t=int(frames[0]);full=m['dof_pos'][t]
    p,nn,_,_=contact_state(chain,full,m['root_pos'][t],rr[t],pad,normal)
    lower=np.array([float(j.find('limit').get('lower')) for j in joints])
    upper=np.array([float(j.find('limit').get('upper')) for j in joints])
    pose=solve_contact_pose(chain,full,m['root_pos'][t],rr[t],pad,normal,p,nn,lower,upper,
                            seeds=[full[ids]+.01],max_nfev=100)
    passed=(max(numerical)<1e-7 and max(fk)<1e-12 and max(batch_error)<1e-12
            and max(coupled_error)<1e-7 and max(unused_error)==0 and pose['contact_feasible'])
    report=dict(status='PASS' if passed else 'FAIL',
                frames=frames.tolist(),central_difference_epsilon_rad=1e-7,
                max_analytic_vs_finite_jacobian_error=float(max(numerical)),
                max_chain_FK_error=float(max(fk)),max_batch_vs_single_error=float(max(batch_error)),
                coupled_17dof_max_finite_jacobian_error=max(coupled_error),
                other_arm_unused_column_max_error=max(unused_error),
                analytic_pose_solver=dict(position_error_m=pose['position_error_m'],
                                          normal_error_rad=pose['normal_error_rad'],
                                          optimizer_success=pose['optimizer_success']))
    return report,(m,plan,urdf,joints,names,chain,rr,ids)


if __name__=='__main__':
    import argparse
    here=Path(__file__).resolve().parent
    sys.path.insert(0,str(here))
    from runtime_guard import install,WORK
    install()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',action='store_true')
    parser.add_argument('--max-nfev',type=int,default=80)
    parser.add_argument('--output-dir',type=Path,default=WORK/'execution/priority_2_hand_contact/solver_tests')
    args=parser.parse_args()
    report,data=_fixtures()
    out=args.output_dir.resolve()
    if WORK not in out.parents:raise ValueError('Output must be under InterPose')
    out.mkdir(parents=True,exist_ok=True)
    fixture_path=out/'analytic_jacobian.json'
    if fixture_path.exists():raise FileExistsError('Do not overwrite prior fixtures')
    fixture_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if report['status']!='PASS':raise SystemExit(1)
    if args.experiment:
        import pickle
        m,plan,urdf,joints,names,chain,rr,ids=data
        source=WORK/'execution/priority_2_hand_contact/attempt02/interpose_floorlamp_g1_hand_contact.pkl'
        with source.open('rb') as f:initial=pickle.load(f)['dof_pos'][:,ids]
        with np.load(source.with_suffix('.npz')) as obj:
            ro=Rotation.from_quat(obj['rot']).as_matrix();op=obj['trans'].copy()
        spec=plan['hands']['left']
        target=op+np.einsum('nij,j->ni',ro,np.array(spec['target_point_object']))
        normal=-np.einsum('nij,j->ni',ro,np.array(spec['outward_normal_object']))
        low=np.array([float(j.find('limit').get('lower')) for j in joints])
        high=np.array([float(j.find('limit').get('upper')) for j in joints])
        limits=np.array([float(j.find('limit').get('velocity')) for j in joints])
        result=smooth_contact_trajectory(chain,m['dof_pos'],m['root_pos'],rr,spec['pad_local'],spec['pad_normal_local'],
                 target,normal,initial,low,high,limits,fps=m['fps'],max_nfev=args.max_nfev,
                 progress=lambda x:print(json.dumps(x),flush=True))
        np.save(out/'left_trajectory.npy',result.pop('q'))
        (out/'trajectory_experiment.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
