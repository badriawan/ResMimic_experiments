"""Whole-trajectory sparse constrained arm smoothing; no per-frame reprojection.

D2/D3 terms act on ALL arm samples in one optimization. They are comparative
smoothness regularizers, not certified actuator acceleration/jerk limits.
"""
import argparse,copy,json,pickle,shutil,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,BASELINE,ASSETS,URDF,install,sha,write_json
install()
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix,diags,eye,kron,vstack
from scipy.sparse.linalg import spsolve,splu
from scipy.spatial.transform import Rotation
from solver_kinematics import Chain
from hand_contact_ik import ArmContactKinematics
from geometry_oracle import Oracle


def stats(x):
    x=np.asarray(x);return dict(rms=float(np.sqrt(np.mean(x*x))),p95_abs=float(np.percentile(abs(x),95)),maximum_abs=float(abs(x).max()))


def derivatives(q,fps):return {name:stats(np.diff(q,n=k,axis=0)*fps**k) for k,name in [(1,'velocity'),(2,'acceleration'),(3,'jerk')]}


class Problem:
    def __init__(self,motion,objects,plan,args):
        self.m=motion;self.o=objects;self.plan=plan;self.args=args;self.q0=motion['dof_pos'][:,15:].copy();self.n,self.d=self.q0.shape;self.fps=float(motion['fps']);self.size=self.n*self.d
        self.robot=ET.parse(URDF).getroot();self.names=[j.get('name') for j in self.robot.findall('joint') if j.get('type')!='fixed'];joints=[j for j in self.robot.findall('joint') if j.get('type')!='fixed']
        self.lower=np.asarray([float(j.find('limit').get('lower')) for j in joints])[15:];self.upper=np.asarray([float(j.find('limit').get('upper')) for j in joints])[15:];self.speed=np.asarray([float(j.find('limit').get('velocity')) for j in joints])[15:]
        rr=Rotation.from_quat(motion['root_rot']).as_matrix();ro=Rotation.from_quat(objects['rot']).as_matrix();self.rr=rr;self.ro=ro
        self.models=[];self.tmodels=[];self.targets=[];self.normal_targets=[];self.tangent_targets=[];self.out=[];self.anchor=[];self.handvertices=[]
        for side in ['left','right']:
            h=plan['hands'][side];chain=Chain(self.robot,self.names,h['link']);ids=np.arange(15,29)
            self.models.append(ArmContactKinematics(chain,motion['dof_pos'],motion['root_pos'],rr,h['pad_local'],h['pad_normal_local'],ids))
            self.tmodels.append(ArmContactKinematics(chain,motion['dof_pos'],motion['root_pos'],rr,h['pad_local'],h['tangent_local'],ids))
            _,baseline_tangent=self.tmodels[-1].evaluate(self.q0,False)
            initial_local=ro[0].T@baseline_tangent[0];normal_local=np.asarray(h['outward_normal_object']);initial_local-=normal_local*np.dot(initial_local,normal_local);initial_local/=np.linalg.norm(initial_local)
            self.tangent_targets.append(np.einsum('tij,j->ti',ro,initial_local))
            out=np.einsum('tij,j->ti',ro,np.asarray(h['outward_normal_object']));anchor=objects['trans']+np.einsum('tij,j->ti',ro,np.asarray(h['target_point_object']))
            self.out.append(out);self.anchor.append(anchor);self.targets.append(anchor+float(h['target_gap_m'])*out);self.normal_targets.append(-out)
            import trimesh
            self.handvertices.append(np.asarray(trimesh.load_mesh(URDF.parent/'meshes'/(h['link']+'.STL'),process=True).convex_hull.vertices))
        self.D1=kron(diags([-np.ones(self.n-1),np.ones(self.n-1)],[0,1],shape=(self.n-1,self.n)),eye(14),format='csr')
        self.D2=kron(diags([np.ones(self.n-2),-2*np.ones(self.n-2),np.ones(self.n-2)],[0,1,2],shape=(self.n-2,self.n)),eye(14),format='csr')
        self.D3=kron(diags([-np.ones(self.n-3),3*np.ones(self.n-3),-3*np.ones(self.n-3),np.ones(self.n-3)],[0,1,2,3],shape=(self.n-3,self.n)),eye(14),format='csr')
        self.temporal=vstack([self.D2*(self.fps**2/args.acceleration_scale),self.D3*(self.fps**3/args.jerk_scale)],format='csr')
        self.reg=eye(self.size,format='csr')*args.reference_weight
        tangent_D2=kron(diags([np.ones(self.n-2),-2*np.ones(self.n-2),np.ones(self.n-2)],[0,1,2],shape=(self.n-2,self.n)),eye(6),format='csr')
        tangent_D3=kron(diags([-np.ones(self.n-3),3*np.ones(self.n-3),-3*np.ones(self.n-3),np.ones(self.n-3)],[0,1,2,3],shape=(self.n-3,self.n)),eye(6),format='csr')
        self.tangent_temporal=vstack([tangent_D2*(self.fps**2/args.tangent_acceleration_scale),tangent_D3*(self.fps**3/args.tangent_jerk_scale)],format='csr')
        self.g=Oracle();self.pairs=self.g.self_pairs+[(name,'floorlamp') for name in self.g.object_nonhand_links]
        self.chains={name:Chain(self.robot,self.names,name) for name in self.m['link_body_list'] if name!='pelvis'}
        self.active={int(j):k for k,j in enumerate(range(15,29))}
        self.sampleframes=np.unique(np.r_[np.arange(0,self.n,args.collision_stride),self.n-1]).astype(int)
        self.contact_rows=18 if args.tangent_weight else 12
        self.blockrows=np.repeat(np.arange(self.n*self.contact_rows),14);self.blockcols=(np.arange(self.n)[:,None,None]*14+np.arange(14)[None,None,:]+np.zeros((1,self.contact_rows,1),int)).ravel()

    def core(self,q,jac=True):
        residuals=[];blocks=[];points=[];normals=[];tangents=[];gap_res=[];gap_jac=[];tangent_blocks=[]
        for i,model in enumerate(self.models):
            if jac:
                p,nn,jp,jn=model.evaluate(q)
                residuals.extend([(p-self.targets[i])*self.args.position_weight,(nn-self.normal_targets[i])*self.args.normal_weight]);blocks.extend([jp*self.args.position_weight,jn*self.args.normal_weight])
            else:
                p,nn=model.evaluate(q,False);residuals.extend([(p-self.targets[i])*self.args.position_weight,(nn-self.normal_targets[i])*self.args.normal_weight])
            if self.args.tangent_weight:
                if jac:
                    _,tangent,_,jt=self.tmodels[i].evaluate(q);blocks.append(jt*self.args.tangent_weight)
                else:_,tangent=self.tmodels[i].evaluate(q,False)
                residuals.append((tangent-self.tangent_targets[i])*self.args.tangent_weight)
                tangents.append(np.einsum('tji,tj->ti',self.ro,tangent))
                if jac:tangent_blocks.append(np.einsum('tji,tjk->tik',self.ro,jt))
            gap=np.einsum('ij,ij->i',p-self.anchor[i],self.out[i]);outside=(gap<self.args.pad_gap_lower)|(gap>self.args.pad_gap_upper)
            gap_res.append((gap-np.clip(gap,self.args.pad_gap_lower,self.args.pad_gap_upper))*self.args.pad_gap_weight)
            if jac:gap_jac.append(np.einsum('ni,nij->nj',self.out[i],jp)*outside[:,None]*self.args.pad_gap_weight)
            points.append(p);normals.append(nn)
        # Contact residual rows frame-major: leftposition, leftnormal, rightposition,rightnormal.
        contact=np.concatenate(residuals,axis=1).ravel();x=q.ravel()
        gapres=np.stack(gap_res,axis=1).ravel()
        residual=np.r_[contact,gapres,self.temporal@x,self.reg@(x-self.q0.ravel())]
        if self.args.tangent_weight:
            tangent_values=np.concatenate(tangents,axis=1).ravel()
            residual=np.r_[residual,self.tangent_temporal@tangent_values]
        if not jac:return residual
        jblock=np.concatenate(blocks,axis=1)
        Jc=coo_matrix((jblock.ravel(),(self.blockrows,self.blockcols)),shape=(self.n*self.contact_rows,self.size)).tocsr();Jc.eliminate_zeros()
        gj=np.stack(gap_jac,axis=1);gr=np.repeat(np.arange(self.n*2),14);gc=(np.arange(self.n)[:,None,None]*14+np.arange(14)[None,None,:]+np.zeros((1,2,1),int)).ravel()
        Jg=coo_matrix((gj.ravel(),(gr,gc)),shape=(self.n*2,self.size)).tocsr();Jg.eliminate_zeros()
        jacobians=[Jc,Jg,self.temporal,self.reg]
        if self.args.tangent_weight:
            tjblock=np.concatenate(tangent_blocks,axis=1);tr=np.repeat(np.arange(self.n*6),14);tc=(np.arange(self.n)[:,None,None]*14+np.arange(14)[None,None,:]+np.zeros((1,6,1),int)).ravel()
            Jt=coo_matrix((tjblock.ravel(),(tr,tc)),shape=(self.n*6,self.size)).tocsr();Jt.eliminate_zeros();jacobians.append(self.tangent_temporal@Jt)
        return residual,vstack(jacobians,format='csr')

    def point_jac(self,link,point,frame,full):
        if link in ['pelvis','floorlamp']:return np.zeros((3,14))
        chain=self.chains[link];p=self.m['root_pos'][frame].copy();r=self.rr[frame].copy();axes=np.zeros((14,3));origins=np.zeros((14,3))
        for xyz,rot,index,axis in chain.steps:
            p=p+r@xyz;r=r@rot
            if index is not None:
                if index in self.active:axes[self.active[index]]=r@axis;origins[self.active[index]]=p
                r=r@Rotation.from_rotvec(full[index]*axis).as_matrix()
        return np.cross(axes,point-origins).T

    def collisions(self,q,jac=False):
        values=[];ridx=[];cidx=[];data=[];worst=1.;bad=0
        for frame in self.sampleframes:
            full=self.m['dof_pos'][frame].copy();full[15:]=q[frame]
            self.g.set_frame(full,self.m['root_pos'][frame],self.m['root_rot'][frame],self.o['trans'][frame],self.o['rot'][frame])
            for a,b in self.pairs:
                value=self.g.fast_pair_distance(a,b,threshold=self.args.clearance+.001);distance=value['distance_m'];worst=min(worst,distance)
                if distance<self.args.clearance:
                    row=len(values);values.append((self.args.clearance-distance)*self.args.collision_weight);bad+=1
                    if jac and value.get('normal_valid',False):
                        gradient=value['normal_a_to_b']@(self.point_jac(b,value['point_b_world'],frame,full)-self.point_jac(a,value['point_a_world'],frame,full))
                        for j,g in enumerate(gradient):
                            if g:ridx.append(row);cidx.append(frame*14+j);data.append(-g*self.args.collision_weight)
            # Actual source hand hulls against the known stem facet; avoid accepting unsigned CCD zero.
            for side in ['left','right']:
                h=self.plan['hands'][side];name=h['link'];out=self.ro[frame]@h['outward_normal_object'];distance=self.g.hand_lamp_distance(name,out)
                d=distance['distance_m'];worst=min(worst,d)
                if d<self.args.hand_min_gap:
                    row=len(values);values.append((self.args.hand_min_gap-d)*self.args.collision_weight);bad+=1
                    if jac and distance.get('normal_valid',False):
                        gradient=distance['normal_a_to_b']@(self.point_jac('floorlamp',distance['point_b_world'],frame,full)-self.point_jac(name,distance['point_a_world'],frame,full))
                        for j,g in enumerate(gradient):
                            if g:ridx.append(row);cidx.append(frame*14+j);data.append(-g*self.args.collision_weight)
        r=np.asarray(values)
        if jac:return r,coo_matrix((data,(ridx,cidx)),shape=(len(values),self.size)).tocsr(),worst,bad
        return r,worst,bad

    def constrained_direction(self,H,gradient,q):
        """Project the Newton direction onto active linear bounds using H-metric.

        Position and velocity constraints are affine in all trajectory samples.
        A small active-set equality solve prevents one saturated source edge
        from forcing a zero global step; no hardware bound is enlarged.
        """
        factor=splu(H,permc_spec='NATURAL');raw=factor.solve(-gradient);direction=raw.copy();rows=[];seen=set()
        for active_iteration in range(8):
            dd=direction.reshape(q.shape);dq=np.diff(q,axis=0);dstep=np.diff(dd,axis=0);cap=self.speed/self.fps*.95
            for frame,joint in np.argwhere(((q-self.lower<1e-7)&(dd< -1e-12))|((self.upper-q<1e-7)&(dd>1e-12))):
                key=('q',int(frame),int(joint))
                if key not in seen:
                    rows.append(({int(frame)*14+int(joint):1.}));seen.add(key)
            for frame,joint in np.argwhere(((dq-cap> -1e-7)&(dstep>1e-12))|((-dq-cap> -1e-7)&(dstep< -1e-12))):
                key=('v',int(frame),int(joint))
                if key not in seen:
                    rows.append({int(frame)*14+int(joint):-1.,(int(frame)+1)*14+int(joint):1.});seen.add(key)
            if not rows:break
            rr=[];cc=[];vv=[]
            for i,row in enumerate(rows):
                for j,value in row.items():rr.append(i);cc.append(j);vv.append(value)
            E=coo_matrix((vv,(rr,cc)),shape=(len(rows),self.size)).tocsr();Z=factor.solve(E.T.toarray());S=np.asarray(E@Z);lagrange=np.linalg.lstsq(S,np.asarray(E@raw),rcond=1e-11)[0];corrected=raw-Z@lagrange
            if np.max(abs(corrected-direction))<1e-11:direction=corrected;break
            direction=corrected
        return direction.reshape(q.shape),len(rows)

    def step_bound(self,q,delta):
        # All accepted steps respect true position limits and the predeclared95% velocity cap.
        near=((q-self.lower)<1e-7)&(delta<0)|((self.upper-q)<1e-7)&(delta>0);delta=delta.copy();delta[near]=0
        alpha=1.
        for x,d,lo,hi in [(q,delta,self.lower,self.upper),(np.diff(q,axis=0),np.diff(delta,axis=0),-self.speed/self.fps*.95-1e-10,self.speed/self.fps*.95+1e-10)]:
            allowed=np.where(d>1e-12,(hi-x)/np.where(d>1e-12,d,1),np.where(d< -1e-12,(lo-x)/np.where(d< -1e-12,d,1),np.inf))
            alpha=min(alpha,float(np.min(allowed)))
        return max(0,min(1,alpha*.995)),delta


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',default='attempt01');p.add_argument('--iterations',type=int,default=15);p.add_argument('--acceleration-scale',type=float,default=300.);p.add_argument('--jerk-scale',type=float,default=150000.);p.add_argument('--position-weight',type=float,default=30000.);p.add_argument('--normal-weight',type=float,default=5000.);p.add_argument('--reference-weight',type=float,default=.5);p.add_argument('--collision-weight',type=float,default=30000.);p.add_argument('--clearance',type=float,default=.002);p.add_argument('--hand-min-gap',type=float,default=.0005);p.add_argument('--collision-stride',type=int,default=4);p.add_argument('--damping',type=float,default=10000.);p.add_argument('--damping-floor',type=float,default=1000.);p.add_argument('--pad-gap-lower',type=float,default=.00115);p.add_argument('--pad-gap-upper',type=float,default=.00185);p.add_argument('--pad-gap-weight',type=float,default=100000.);p.add_argument('--seed',type=Path);p.add_argument('--tangent-weight',type=float,default=0.);p.add_argument('--tangent-acceleration-scale',type=float,default=100.);p.add_argument('--tangent-jerk-scale',type=float,default=50000.);args=p.parse_args();started=time.monotonic()
    out=ROOT/'candidates'/args.name;out.mkdir(parents=True,exist_ok=False)
    for folder in ['reports','logs']:(ROOT/folder).mkdir(exist_ok=True)
    m=pickle.load(BASELINE.open('rb'));objects={k:v.copy() for k,v in np.load(BASELINE.with_suffix('.npz'),allow_pickle=False).items()};plan=json.loads((BASELINE.parent/'contact_plan.json').read_text());problem=Problem(m,objects,plan,args);q=problem.q0.copy() if args.seed is None else pickle.load(args.seed.open('rb'))['dof_pos'][:,15:].copy();trace=[]
    print('Baseline',json.dumps(derivatives(q,problem.fps)),flush=True)
    damping=args.damping
    for iteration in range(args.iterations):
        r,J=problem.core(q);rc,Jc,worst,bad=problem.collisions(q,True);rall=np.r_[r,rc];Jall=vstack([J,Jc],format='csr');cost=float(rall@rall/2)
        H=(Jall.T@Jall+eye(problem.size,format='csc')*damping).tocsc();gradient=np.asarray(Jall.T@rall).ravel()
        delta,active_constraints=problem.constrained_direction(H,gradient,q);alpha,delta=problem.step_bound(q,delta)
        accepted=False
        for trial in range(9):
            if alpha<1e-8:break
            candidate=q+alpha*delta;rr=problem.core(candidate,False);rrc,ww,bb=problem.collisions(candidate,False);newcost=float((rr@rr+rrc@rrc)/2)
            if np.isfinite(newcost) and newcost<cost:
                q=candidate;accepted=True;damping=max(args.damping_floor,damping*.5);break
            alpha*=.5
        if not accepted:damping*=10;newcost=cost;ww=worst;bb=bad
        row=dict(iteration=iteration,active_linear_constraints=active_constraints,cost_before=cost,cost_after=newcost,alpha=alpha,accepted=accepted,damping=damping,maximum_step_rad=float(abs(alpha*delta).max()),minimum_sampled_distance_m=float(ww),collision_hinge_count=int(bb),elapsed_s=time.monotonic()-started,metrics=derivatives(q,problem.fps));trace.append(row);np.savez_compressed(out/'solver_checkpoint.npz',q=q,iteration=iteration,trace_json=np.asarray(json.dumps(trace)));print(json.dumps(row),flush=True)
        if accepted and abs(cost-newcost)<cost*1e-6:break
        if not accepted and damping>1e8:break
    result=copy.deepcopy(m);result['dof_pos'][:,15:]=q
    sys.path.insert(0,str(REPO/'legged_gym/legged_gym/scripts'));from convert_gmr_qpos import forward_kinematics
    result['local_body_pos'],_=forward_kinematics(problem.robot,result['dof_pos'],problem.names,result['link_body_list'])
    filename=out/'interpose_floorlamp_g1_hand_contact.pkl'
    with filename.open('wb') as f:pickle.dump(result,f,protocol=4)
    shutil.copyfile(BASELINE.with_suffix('.npz'),filename.with_suffix('.npz'));shutil.copyfile(BASELINE.parent/'contact_plan.json',out/'contact_plan.json')
    pe=[];ne=[]
    for i,model in enumerate(problem.models):
        point,normal=model.evaluate(q,False);pe.append(np.linalg.norm(point-problem.targets[i],axis=1));ne.append(np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',normal,problem.normal_targets[i]),-1,1))))
    gates=dict(root_exact=np.array_equal(result['root_pos'],m['root_pos']),root_rotation_exact=np.array_equal(result['root_rot'],m['root_rot']),first15_exact=np.array_equal(result['dof_pos'][:,:15],m['dof_pos'][:,:15]),object_byte_exact=sha(filename.with_suffix('.npz'))==sha(BASELINE.with_suffix('.npz')),joint_positions=bool(np.all(q>=problem.lower-1e-8) and np.all(q<=problem.upper+1e-8)),joint_velocity=bool(np.max(abs(np.diff(q,axis=0))*problem.fps/problem.speed)<=.95+1e-7),contact_point_2mm=max(float(v.max()) for v in pe)<=.002,contact_normal_5deg=max(float(v.max()) for v in ne)<=5)
    report=dict(status='SMOOTHING_CANDIDATE_PENDING_INDEPENDENT_GEOMETRY' if all(gates.values()) else 'FAIL_NATIVE_SMOOTHING_GATES',gates=gates,method='Whole-trajectory sparseGaussNewton analyticcontactJac+globalD2D3+periodicsourcecolliderhinges; no perframefinalreprojection',frames=problem.n,fps=problem.fps,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},regularizer_interpretation='Acceleration/jerk scales normalize objective terms; comparative goals, not hardware limits',before=derivatives(problem.q0,problem.fps),after=derivatives(q,problem.fps),maximum_joint_departure_rad=float(abs(q-problem.q0).max()),contact_point_max_m=[float(v.max()) for v in pe],contact_normal_max_deg=[float(v.max()) for v in ne],iterations=trace,source_sha256={str(p):sha(p) for p in [BASELINE,BASELINE.with_suffix('.npz'),BASELINE.parent/'contact_plan.json',URDF]},outputs={str(p):sha(p) for p in [filename,filename.with_suffix('.npz'),out/'contact_plan.json']},training_executed=False,policy_executed=False,object_path_frozen=True,root_leg_waist_frozen=True)
    write_json(out/'solver_report.json',report);write_json(ROOT/('reports/solver_'+args.name+'.json'),report);print(json.dumps({k:report[k] for k in ['status','gates','before','after','maximum_joint_departure_rad','contact_point_max_m','contact_normal_max_deg']}),flush=True)

if __name__=='__main__':main()
