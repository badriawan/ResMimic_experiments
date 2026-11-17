"""Conditional friction-wrench analysis of corrected reference; no dynamic simulation."""
import argparse,json,sys,pickle,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,ASSETS,URDF,install,sha,write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidate',type=Path,required=True);args=parser.parse_args();install()
    path=args.candidate.resolve()
    if ROOT not in path.parents:raise ValueError('Candidate must be inside step_2')
    import numpy as np
    import mujoco,trimesh
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    from scipy.optimize import linprog
    sys.path.insert(1,str(REPO/'legged_gym/legged_gym/scripts'))
    from render_motion import build_scene,set_frame
    with path.open('rb') as handle:motion=pickle.load(handle)
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:obj={k:z[k].copy() for k in z.files}
    plan=json.loads((path.parent/'contact_plan.json').read_text());profile=json.loads((ASSETS/'asset_profile.json').read_text())
    model,addresses,rootadr,objid=build_scene(URDF,ASSETS/'floorlamp_visual.obj',960,720);data=mujoco.MjData(model)
    handids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,side+'_rubber_hand') for side in ['left','right']]
    jointnames=[joint.get('name') for joint in ET.parse(URDF).getroot().findall('joint') if joint.get('type')=='revolute']
    jids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name) for name in jointnames];dofadr=model.jnt_dofadr[jids]
    # Recorded effective torque limits, same robot/controller as step1. They are diagnostic, not proof of available net torque.
    torque_source=ROOT.parents[1]/'step_1_walking/results/attempt02/rollout_states.npz'
    with np.load(torque_source,allow_pickle=False) as z:limits=z['torque_limits'].copy()
    stem=next(component for component in profile['components'] if component['name']=='stem')
    hull=ConvexHull(np.asarray(trimesh.load_mesh(stem['collision_mesh'],process=False).vertices));equations=hull.equations
    mass=float(profile['total_mass_kg']);mu=float(profile['friction']);comlocal=np.asarray(profile['com_m']);inertia=np.asarray(profile['inertia_kg_m2']);fps=float(motion['fps']);dt=1/fps;n=len(motion['root_pos'])
    rotation=Rotation.from_quat(obj['rot']).as_matrix();acc=np.gradient(np.gradient(obj['trans'],dt,axis=0,edge_order=2),dt,axis=0,edge_order=2)
    relative=Rotation.from_matrix(rotation[1:]@np.swapaxes(rotation[:-1],1,2)).as_rotvec()/dt
    omega=np.vstack([relative[:1],(relative[:-1]+relative[1:])/2,relative[-1:]])
    alpha=np.gradient(omega,dt,axis=0,edge_order=2)
    comworld=obj['trans']+np.einsum('tij,j->ti',rotation,comlocal)
    # Rigid-body COM acceleration includes rotating offset from object root.
    offset=np.einsum('tij,j->ti',rotation,comlocal)
    acom=acc+np.cross(alpha,offset)+np.cross(omega,np.cross(omega,offset))
    iw=rotation@inertia@np.swapaxes(rotation,1,2)
    dynamic_force=mass*(acom-np.array([0,0,-9.81]));dynamic_moment=np.einsum('tij,tj->ti',iw,alpha)+np.cross(omega,np.einsum('tij,tj->ti',iw,omega))
    def clip_polygon(polygon,normal,bias):
        if not len(polygon):return polygon
        result=[]
        for i,a in enumerate(polygon):
            b=polygon[(i+1)%len(polygon)];da=normal@a+bias;db=normal@b+bias;ina=da<=1e-9;inb=db<=1e-9
            if ina:result.append(a)
            if ina!=inb:result.append(a+(b-a)*da/(da-db))
        return np.asarray(result)
    def patch(frame,hand,R,pos):
        hp=plan['hands'][hand];tri=np.asarray(hp['pad_triangle_local'])@R.T+pos
        local=(tri-obj['trans'][frame])@rotation[frame]
        normal=np.asarray(hp['outward_normal_object'],float);normal/=np.linalg.norm(normal)
        face=int(np.argmax(equations[:,:3]@normal));eq=equations[face];normal=eq[:3];signed=local@normal+eq[3]
        polygon=local-signed[:,None]*normal
        for plane in equations:polygon=clip_polygon(polygon,plane[:3],plane[3])
        if len(polygon)<3:return np.empty((0,3)),float(abs(signed).max()),0.
        area=.5*np.linalg.norm(sum(np.cross(polygon[(j+1)%len(polygon)]-polygon[0],polygon[j]-polygon[0]) for j in range(1,len(polygon)-1)))
        return polygon@rotation[frame].T+obj['trans'][frame],float(abs(signed).max()),float(area)
    def grasp_matrix(points,normals,center):
        columns=[];owners=[]
        for owner,(pp,nn) in enumerate(zip(points,normals)):
            inward=-nn;basis=np.eye(3)[np.argmin(abs(inward))];t1=np.cross(inward,basis);t1/=np.linalg.norm(t1);t2=np.cross(inward,t1)
            for point in pp:
                for angle in np.arange(8)*2*np.pi/8:
                    force=inward+mu*(np.cos(angle)*t1+np.sin(angle)*t2);columns.append(np.r_[force,np.cross(point-center,force)]);owners.append(owner)
        return np.asarray(columns).T,np.asarray(owners)
    def solve(A,wrench,owners,T=None,bias=None):
        if A.shape[0]!=6 or A.shape[1]==0:return dict(exact=False,rank=0,residual_force_N=float(np.linalg.norm(wrench[:3])),residual_moment_Nm=float(np.linalg.norm(wrench[3:])),normal_sum_N=None),None
        answer=linprog(np.ones(A.shape[1]),A_eq=A,b_eq=wrench,bounds=(0,None),method='highs')
        exact=bool(answer.success);x=answer.x if exact else None
        if x is None:
            # Find minimum weighted L1 wrench residual with a small normal-force tie-breaker.
            scale=np.array([1,1,1,5,5,5]);B=A*scale[:,None];target=wrench*scale
            relaxed=linprog(np.r_[np.full(A.shape[1],1e-8),np.ones(12)],A_eq=np.c_[B,np.eye(6),-np.eye(6)],b_eq=target,bounds=(0,None),method='highs')
            if relaxed.success:x=relaxed.x[:A.shape[1]]
        residual=A@x-wrench if x is not None else -wrench
        bounded=linprog(np.ones(A.shape[1]),A_eq=A,b_eq=wrench,A_ub=np.r_[T,-T],b_ub=np.r_[limits,limits],bounds=(0,None),method='highs') if T is not None else None
        gravity_bounded=linprog(np.ones(A.shape[1]),A_eq=A,b_eq=wrench,A_ub=np.r_[-T,T],b_ub=np.r_[limits-bias,limits+bias],bounds=(0,None),method='highs') if T is not None and bias is not None else None
        return dict(exact=exact,rank=int(np.linalg.matrix_rank(A,tol=1e-8)),residual_force_N=float(np.linalg.norm(residual[:3])),residual_moment_Nm=float(np.linalg.norm(residual[3:])),normal_sum_N=float(x.sum()) if x is not None else None,normal_by_hand_N=[float(x[owners==i].sum()) for i in range(2)] if x is not None else None,gravity_plus_contact_torque_bounded_equilibrium_feasible=bool(gravity_bounded.success) if gravity_bounded is not None else None,gravity_bounded_normal_sum_N=float(gravity_bounded.x.sum()) if gravity_bounded is not None and gravity_bounded.success else None,contact_torque_bounded_equilibrium_feasible=bool(bounded.success) if bounded is not None else None,bounded_solution_normal_sum_N=float(bounded.x.sum()) if bounded is not None and bounded.success else None),x
    with np.load(ASSETS/'collider_vertices.npz',allow_pickle=False) as z:floor_vertices=z['vertices'].copy()
    records=[];contact_tau=[];samples=[]
    for frame in range(n):
        set_frame(model,data,motion,obj,frame,addresses,rootadr,objid)
        points=[];patches=[];normals=[];patchgaps=[];patchareas=[]
        for side,bid in zip(['left','right'],handids):
            hp=plan['hands'][side];points.append((rotation[frame]@np.asarray(hp['target_point_object'])+obj['trans'][frame])[None]);normals.append(rotation[frame]@np.asarray(hp['outward_normal_object']))
            polygon,gap,area=patch(frame,side,data.xmat[bid].reshape(3,3),data.xpos[bid]);patches.append(polygon);patchgaps.append(gap);patchareas.append(area)
        gap=float((floor_vertices@rotation[frame].T+obj['trans'][frame])[:,2].min())
        row=dict(frame=frame,time_s=frame*dt,object_floor_gap_m=gap,reference_airborne=bool(gap>.02),patch_projection_max_plane_gap_m=patchgaps,projected_patch_area_m2=patchareas)
        for label,pp in [('point',points),('projected_patch',patches)]:
            A,owners=grasp_matrix(pp,normals,comworld[frame]);A=A if A.size else np.empty((6,0))
            T=np.zeros((29,A.shape[1]));cursor=0
            for owner,vertices in enumerate(pp):
                for point in vertices:
                    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mujoco.mj_jac(model,data,jp,jr,point,handids[owner])
                    T[:,cursor:cursor+8]=-jp[:,dofadr].T@A[:3,cursor:cursor+8];cursor+=8
            for mode,wrench in [('static',np.r_[0,0,mass*9.81,0,0,0]),('dynamic',np.r_[dynamic_force[frame],dynamic_moment[frame]])]:
                result,x=solve(A,wrench,owners,T,data.qfrc_bias[dofadr].copy());row[label+'_'+mode]=result
                if label=='projected_patch' and mode=='dynamic' and x is not None:
                    tau=np.zeros(model.nv);cursor=0
                    for owner,vertices in enumerate(pp):
                        for point in vertices:
                            # Eight columns per vertex, their resultant force on object; opposite load acts on robot.
                            force=A[:3,cursor:cursor+8]@x[cursor:cursor+8];cursor+=8
                            jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mujoco.mj_jac(model,data,jp,jr,point,handids[owner]);tau-=jp.T@force
                    contact_tau.append(tau[dofadr])
        records.append(row)
        if frame%100==0:print('Wrench frame',frame,flush=True)
    def summarize(label):
        values=[row[label] for row in records];forces=[v['normal_sum_N'] for v in values if v['normal_sum_N'] is not None]
        return dict(exact_equilibrium_frames=sum(v['exact'] for v in values),contact_only_torque_bounded_equilibrium_frames=sum(v.get('contact_torque_bounded_equilibrium_feasible',False) for v in values),gravity_plus_contact_bounded_equilibrium_frames=sum(v.get('gravity_plus_contact_torque_bounded_equilibrium_feasible',False) for v in values),frames=n,max_force_residual_N=max(v['residual_force_N'] for v in values),max_moment_residual_Nm=max(v['residual_moment_Nm'] for v in values),reference_airborne_frames=sum(row['reference_airborne'] for row in records),airborne_gravity_plus_contact_bounded_frames=sum(row['reference_airborne'] and row[label].get('gravity_plus_contact_torque_bounded_equilibrium_feasible',False) for row in records),maximum_minimum_total_normal_N=max(forces) if forces else None,grasp_matrix_rank_range=[min(v['rank'] for v in values),max(v['rank'] for v in values)])
    ta=np.asarray(contact_tau);ratios=np.abs(ta)/limits if ta.size else np.empty((0,29))
    report=dict(status='CONDITIONAL_WRENCH_ASSESSMENT',candidate=str(path),mass_kg=mass,friction=mu,com_object_m=comlocal.tolist(),inertia_object_kg_m2=inertia.tolist(),summaries={label:summarize(label) for label in ['point_static','point_dynamic','projected_patch_static','projected_patch_dynamic']},maximum_object_com_acceleration_m_s2=float(np.linalg.norm(acom,axis=1).max()),maximum_object_angular_acceleration_rad_s2=float(np.linalg.norm(alpha,axis=1).max()),contact_only_torque=dict(maximum_effective_limit_ratio=float(ratios.max()) if ratios.size else None,per_joint_peak_ratio=ratios.max(axis=0).tolist() if ratios.size else None,joint_names=jointnames),minimum_projected_patch_area_m2=np.asarray([r['projected_patch_area_m2'] for r in records]).min(axis=0).tolist(),per_frame=records,static_gravity_model=dict(gravity_m_s2=model.opt.gravity.tolist(),robot_mass_kg=float(model.body_mass[model.body_mocapid<0].sum()),actuator_sign='tau_required=qfrc_bias-(-Jtranspose*force_on_object)',scope='URDFstaticgravitypluscontactonly; no robot inertial/ground-reaction inverse dynamics'),limitations=['Static/dynamic LP is conditional on maintained contacts and assumed friction; it does not execute grasp dynamics.','Point model uses fixed actual collider anchors. Opposed point contacts may have rank5 and cannot certify6Dforce closure.','Projected patch is intersection of projected real hand-hull triangle with finite convex stem; assumes that distributed contact can form after closing any gap/alignment error. It is NOT a measured pressure patch or guaranteed torsional capacity.','No upper normal-force budget assumed; minimum required normal forces and contact-only joint torque are diagnostic, not measured grip force.','A separate LP constrains contact-only joint torque to recorded effective limits; this ignores robot gravity/inertia and tracking. Neither feasible nor infeasible alone is a full inverse-dynamics certificate.', 'Reported peak contact-only torque ratio is for the selected unbounded minimum-normal-force solution, not proof that every distribution needs that ratio.','Object derivatives use the candidate native-rate finite differences with one-sided endpoint rules, not measured forces or a continuous-acceleration certificate for the piecewise-linear loader.', 'Static-gravity torque LP uses qvel=0 qfrc_bias; it excludes robot inertia and ground-reaction support. The full-body balance problem remains untested.'],input_sha256={str(p):sha(p) for p in [path,path.with_suffix('.npz'),path.parent/'contact_plan.json',ASSETS/'asset_profile.json',URDF,torque_source]})
    output=ROOT/'reports'/('wrench_'+path.parent.name+'.json');write_json(output,report)
    if ta.size:np.savez_compressed(output.with_suffix('.npz'),contact_only_joint_torque=ta,torque_limits=limits,object_com_acceleration=acom,object_angular_acceleration=alpha)
    print(json.dumps(report['summaries'],indent=2))

if __name__=='__main__':main()
