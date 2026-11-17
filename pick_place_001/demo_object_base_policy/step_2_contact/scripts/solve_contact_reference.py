"""Coupled fixed-hand/lamp reference construction; no policy or dynamics run."""
import argparse,copy,json,pickle,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,SOURCE,MOTION,OBJECT,ASSETS,URDF,install,sha,write_json
install()
import numpy as np,trimesh
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
from solver_kinematics import Chain
sys.path.insert(0,str(REPO/'InterPose/scripts'))
from hand_contact_ik import ArmContactKinematics
sys.path.insert(0,str(REPO/'legged_gym/legged_gym/scripts'))
from convert_gmr_qpos import forward_kinematics

def unit(v):return np.asarray(v)/np.linalg.norm(v)
def stats(v):
    v=np.asarray(v);return dict(min=float(v.min()),median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(v.max()))

def hand_patch(side,mode='upper'):
    path=REPO/'assets/g1/meshes'/f'{side}_rubber_hand.STL';hull=trimesh.load_mesh(path).convex_hull
    eligible=np.flatnonzero(hull.face_normals[:,2]>.98) if mode=='upper' else np.flatnonzero(hull.face_normals[:,1]*(-1 if side=='left' else 1)>.85);face=int(eligible[np.argmax(hull.area_faces[eligible])]);normal=hull.face_normals[face].copy();pad=hull.triangles_center[face].copy()
    tangent=unit(np.array([1.,0,0])-normal*normal[0])
    return dict(link=f'{side}_rubber_hand',pad_local=pad.tolist(),pad_normal_local=normal.tolist(),tangent_local=tangent.tolist(),hull_triangle_vertices=hull.triangles[face].tolist(),pad_triangle_local=hull.triangles[face].tolist(),hull_triangle_area_m2=float(hull.area_faces[face]),mesh_sha256=sha(path),hull_support_plane_max_violation_m=float(((hull.vertices-pad)@normal).max()),support_description='Actual '+mode+' convex-hull triangle centroid, its outward normal, and fixed tangent; finite patch geometry, not proven pressure distribution')

def make_targets(base,obj,args,profile):
    n=len(base['root_pos']);fps=float(base['fps']);rootr=Rotation.from_quat(base['root_rot']).as_matrix()
    u=ET.parse(URDF).getroot();jnames=[j.get('name') for j in u.findall('joint') if j.get('type')!='fixed'];local,lr=forward_kinematics(u,base['dof_pos'],jnames,base['link_body_list']);torso_rotation=rootr@lr['torso_link']
    yaw=np.unwrap(Rotation.from_matrix(torso_rotation).as_euler('xyz')[:,2]);yaw=gaussian_filter1d(yaw,args.sigma,mode='nearest')
    ort=Rotation.from_euler('z',yaw[:,None]).as_matrix();oq=Rotation.from_matrix(ort).as_quat()
    xy=base['root_pos'].copy()+np.einsum('tij,j->ti',ort,[args.forward,args.lateral,0.])
    xy[:,:2]=gaussian_filter1d(xy[:,:2],args.sigma,axis=0,mode='nearest')
    # Keep source lift character but remove frame-scale height changes. Lamp upright: every
    # reference frame must be above the floor, not just initialization.
    source_lift=np.maximum(obj['trans'][:,2]-obj['trans'][:40,2].min(),0)
    lift=gaussian_filter1d(source_lift,args.sigma,mode='nearest');lift-=min(lift[0],lift[-1]);lift=np.maximum(lift,0)
    floor_height=-float(profile['source_bounds_m'][0][2])+.002
    op=xy;op[:,2]=floor_height+lift
    component=next(p for p in profile['components'] if p['name']=='stem');stem=trimesh.load_mesh(ASSETS/Path(component['collision_mesh']).name,process=False)
    normals=stem.face_normals;facets=np.where(abs(normals[:,2])<1e-8)[0];pick=facets[np.argmax(normals[facets,1])];out=unit(normals[pick]);center=np.r_[component['center_xy_m'],0.]
    plane=float(np.max(stem.vertices@out));radius=plane-float(center@out)
    hands={};points=[];ns=[];tangents=[]
    for index,side in enumerate(['left','right']):
        h=hand_patch(side,args.patch);normal=out if side=='left' else -out
        height=args.grip_height+(args.stagger if side=='left' else -args.stagger)
        com_offset=np.asarray(profile['com_m'])-center;com_tangent=com_offset-normal*np.dot(com_offset,normal);com_tangent[2]=0.
        p=center+normal*radius+com_tangent;p[2]=height
        h.update(target_point_object=p.tolist(),outward_normal_object=normal.tolist(),target_gap_m=args.gap,target_tangent_object=unit(np.array([1.,0.,0.])-normal*normal[0]).tolist(),grip_height_object_m=height)
        hands[side]=h;points.append(op+np.einsum('tij,j->ti',ort,p+normal*args.gap));ns.append(-np.einsum('tij,j->ti',ort,normal));tangents.append(np.einsum('tij,j->ti',ort,unit(np.array([1.,0.,0.])-normal*normal[0])))
    return op,oq,np.stack(points,axis=1),np.stack(ns,axis=1),np.stack(tangents,axis=1),hands

def solve(base,points,normals,tangents,hands,args):
    u=ET.parse(URDF).getroot();joints=[j for j in u.findall('joint') if j.get('type')!='fixed'];names=[j.get('name') for j in joints]
    lower=np.array([float(j.find('limit').get('lower')) for j in joints]);upper=np.array([float(j.find('limit').get('upper')) for j in joints]);velocity=np.array([float(j.find('limit').get('velocity')) for j in joints]);n=len(points);rr=Rotation.from_quat(base['root_rot']).as_matrix();q=base['dof_pos'].copy();trace=[]
    for si,side in enumerate(['left','right']):
        h=hands[side];chain=Chain(u,names,h['link']);ids=np.array(chain.indices[-7:]);previous=base['dof_pos'][0,ids].copy()
        for frame in range(n):
            reference=base['dof_pos'][frame,ids]
            kin=ArmContactKinematics(chain,base['dof_pos'][frame:frame+1],base['root_pos'][frame:frame+1],rr[frame:frame+1],h['pad_local'],h['pad_normal_local'],ids)
            tangent=ArmContactKinematics(chain,base['dof_pos'][frame:frame+1],base['root_pos'][frame:frame+1],rr[frame:frame+1],h['pad_local'],h['tangent_local'],ids)
            cache={}
            def calc(x):
                if 'x' not in cache or not np.array_equal(cache['x'],x):
                    p,normal,jp,jn=kin.evaluate(x[None]);_,t,junk,jt=tangent.evaluate(x[None]);cache.update(x=x.copy(),p=p[0],normal=normal[0],t=t[0],jp=jp[0],jn=jn[0],jt=jt[0])
                return cache
            def fun(x):
                z=calc(x)
                return np.r_[(z['p']-points[frame,si])*2000,(z['normal']-normals[frame,si])*300,(z['t']-tangents[frame,si])*.2,(x-reference)*.05,(x-previous)*.2]
            def jac(x):
                z=calc(x);return np.vstack([z['jp']*2000,z['jn']*300,z['jt']*.2,np.eye(7)*.05,np.eye(7)*.2])
            seeds=[previous]
            if frame==0:
                neutral=np.array([-.4,.45 if side=='left' else -.45,0,1.2,0,0,0]);seeds.extend([neutral,reference])
                rng=np.random.default_rng(1234+si)
                for trial in range(16):
                    seed=reference.copy();seed[:4]=[-.5+rng.normal(0,.35),(.4+rng.normal(0,.15))*(1 if side=='left' else -1),rng.normal(0,.5),rng.uniform(.3,1.6)];seed[4:]=[rng.uniform(-1.7,1.7),rng.uniform(-.7,.7),rng.uniform(-.7,.7)];seeds.append(seed)
            candidates=[]
            for seed in seeds:
                result=least_squares(fun,np.clip(seed,lower[ids]+1e-9,upper[ids]-1e-9),jac=jac,bounds=(lower[ids],upper[ids]),max_nfev=180 if frame==0 else 80,ftol=1e-10,xtol=1e-10,gtol=1e-8)
                val=calc(result.x);pe=np.linalg.norm(val['p']-points[frame,si]);ne=np.degrees(np.arccos(np.clip(val['normal']@normals[frame,si],-1,1)));te=np.degrees(np.arccos(np.clip(val['t']@tangents[frame,si],-1,1)))
                candidates.append(((pe>.002 or ne>2,pe+ne*.001+te*.001 if pe>.002 or ne>2 else 0.,float(np.linalg.norm(result.x-previous))+.01*float(np.linalg.norm(result.x-reference))),result,pe,ne,te))
            _,result,pe,ne,te=min(candidates,key=lambda x:x[0]);q[frame,ids]=result.x;previous=result.x
            trace.append(dict(frame=frame,side=side,point_error_m=float(pe),normal_error_deg=float(ne),tangent_error_deg=float(te),nfev=int(result.nfev)))
            if frame%100==0:print(side,frame,'error',pe,ne,te,flush=True)
    return q,trace,names,u,lower,upper,velocity

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',default='attempt01');p.add_argument('--forward',type=float,default=.32);p.add_argument('--lateral',type=float,default=0.);p.add_argument('--grip-height',type=float,default=.15);p.add_argument('--stagger',type=float,default=0.);p.add_argument('--sigma',type=float,default=6.);p.add_argument('--gap',type=float,default=.0005);p.add_argument('--coupled-seed',type=Path);p.add_argument('--tangent-weight',type=float,default=0.);p.add_argument('--lock-initial-tangent',action='store_true');p.add_argument('--patch',choices=['upper','side'],default='upper');p.add_argument('--self-seed',action='store_true');p.add_argument('--temporal-tag');args=p.parse_args();started=time.monotonic()
    out=ROOT/'candidates'/args.name;out.mkdir(parents=True,exist_ok=False)
    with MOTION.open('rb') as f:base=pickle.load(f)
    with np.load(OBJECT,allow_pickle=False) as z:obj={k:z[k].copy() for k in z.files}
    profile=json.loads((ASSETS/'asset_profile.json').read_text())
    if args.temporal_tag:
        base=pickle.load((ROOT/'reports'/('solver_temporal_'+args.temporal_tag+'_input.pkl')).open('rb'))
        with np.load(ROOT/'reports'/('solver_temporal_'+args.temporal_tag+'_target.npz'),allow_pickle=False) as z:obj={k:z[k].copy() for k in z.files}
        op,oq=obj['trans'].copy(),obj['rot'].copy();ro=Rotation.from_quat(oq).as_matrix();hands=copy.deepcopy(json.loads((ROOT/'candidates/attempt11/contact_plan.json').read_text())['hands'])
        points=np.stack([op+np.einsum('tij,j->ti',ro,np.array(hands[s]['target_point_object'])+np.array(hands[s]['outward_normal_object'])*args.gap) for s in ['left','right']],axis=1);normals=np.stack([-np.einsum('tij,j->ti',ro,np.array(hands[s]['outward_normal_object'])) for s in ['left','right']],axis=1);tangents=np.stack([np.einsum('tij,j->ti',ro,np.array(hands[s]['target_tangent_object'])) for s in ['left','right']],axis=1)
    else:op,oq,points,normals,tangents,hands=make_targets(base,obj,args,profile)
    if args.coupled_seed or args.self_seed or args.temporal_tag:
        if args.temporal_tag:seed=dict(dof_pos=base['dof_pos'].copy())
        elif args.self_seed:
            initial_q,initial_trace,*_=solve(base,points,normals,tangents,hands,args);seed=dict(dof_pos=initial_q)
            write_json(out/'solver_seed_trace.json',initial_trace)
        else:seed=pickle.load(args.coupled_seed.open('rb'))
        u=ET.parse(URDF).getroot();joints=[j for j in u.findall('joint') if j.get('type')!='fixed'];names=[j.get('name') for j in joints];lower=np.array([float(j.find('limit').get('lower')) for j in joints]);upper=np.array([float(j.find('limit').get('upper')) for j in joints]);velocity=np.array([float(j.find('limit').get('velocity')) for j in joints])
        from solver_coupled import refine_coupled
        q,op,oq,trace=refine_coupled(base,seed['dof_pos'],op,oq,hands,names,u,lower,upper,velocity,float(base['fps']),args.tangent_weight,args.lock_initial_tangent,bool(args.temporal_tag))
        ro=Rotation.from_quat(oq).as_matrix();points=np.stack([op+np.einsum('tij,j->ti',ro,np.array(hands[s]['target_point_object'])+np.array(hands[s]['outward_normal_object'])*args.gap) for s in ['left','right']],axis=1);normals=np.stack([-np.einsum('tij,j->ti',ro,np.array(hands[s]['outward_normal_object'])) for s in ['left','right']],axis=1)
    else:q,trace,names,u,lower,upper,velocity=solve(base,points,normals,tangents,hands,args)
    m=copy.deepcopy(base);m['dof_pos']=q;m['local_body_pos'],rots=forward_kinematics(u,q,names,m['link_body_list']);obj['trans']=op;obj['rot']=oq
    rootr=Rotation.from_quat(m['root_rot']).as_matrix();objr=Rotation.from_quat(oq).as_matrix()
    for side in ['left','right']:
        hand=hands[side];actual_tangent=np.einsum('tij,j->ti',rootr@rots[hand['link']],np.array(hand['tangent_local']));target_tangent=np.einsum('tij,j->ti',objr,np.array(hand['target_tangent_object']));angle=np.degrees(np.arccos(np.clip(np.einsum('ti,ti->t',actual_tangent,target_tangent),-1,1)))
        for row in trace:
            if row['side']==side:row['tangent_error_deg']=float(angle[row['frame']])
    path=out/'interpose_floorlamp_g1_hand_contact.pkl'
    with path.open('xb') as f:pickle.dump(m,f,protocol=4)
    np.savez_compressed(path.with_suffix('.npz'),**obj)
    np.savez_compressed(out/'solver_targets.npz',point_world=points,normal_world=normals,tangent_world=np.stack([np.einsum('tij,j->ti',objr,np.asarray(hands[side]['target_tangent_object'])) for side in ['left','right']],axis=1))
    speeds=np.abs(np.diff(q,axis=0))*base['fps'];position=np.array([row['point_error_m'] for row in trace]);normal=np.array([row['normal_error_deg'] for row in trace]);tangent=np.array([row['tangent_error_deg'] for row in trace]);active=np.arange(15,29)
    gates=dict(root_position_exact=np.array_equal(m['root_pos'],base['root_pos']),root_rotation_exact=np.array_equal(m['root_rot'],base['root_rot']),legs_waist_exact=np.array_equal(q[:,:15],base['dof_pos'][:,:15]),joint_position_limits=bool(((q>=lower-1e-8)&(q<=upper+1e-8)).all()),joint_speed_limits=bool((speeds<=velocity+1e-6).all()),contact_point_2mm=bool(position.max()<=.002),contact_normal_2deg=bool(normal.max()<=2),finite=bool(np.isfinite(q).all()))
    plan=dict(strategy='Pregrasped opposing fixed-palm facets on actual conservative stem collider, fixed material contact centers and opposing normals, free in-plane hand twist with weak forward bias; no active gripper, no approach/pickup certification',free_tangential_twist=True,contact_from_start=True,hold_start_frame=0,contact_start_frame=0,contact_end_frame=len(q)-1,contact_phase_frames=[0,len(q)-1],active_joint_names=names[15:],invariant_joint_names=names[:15],hands=hands,object_path=dict(horizontal='Filtered torso-heading anterior offset, independent of failed source hand contact',forward_m=args.forward,lateral_m=args.lateral,orientation='Upright with smoothed torso heading',height='Source lift envelope smoothed, global floor clearance2mm',filter_sigma_frames=args.sigma,initial_position_delta_m=(op[0]-np.load(OBJECT)['trans'][0]).tolist()),frames=len(q),fps=base['fps'],solver_arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},coupled_object_xy_yaw_optimized=bool((args.coupled_seed or args.self_seed) and not args.temporal_tag),object_path_frozen_during_reprojection=bool(args.temporal_tag),temporal_model_report=str(ROOT/'reports'/('solver_temporal_'+args.temporal_tag+'.json')) if args.temporal_tag else None,physical_grasp_verified=False,policy_run=False,training_run=False)
    report=dict(status='PASS_KINEMATIC_SOLVER_PENDING_GEOMETRY' if all(gates.values()) else 'FAIL_KINEMATIC_SOLVER',gates=gates,position_error_m=stats(position),normal_error_deg=stats(normal),tangent_error_deg=stats(tangent),speed_ratio_max=float((speeds/velocity).max()),solver_arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},solver_implementation_sha256={str(path):sha(path) for path in (ROOT/'scripts').glob('solver_*.py')},max_active_joint_change_rad=float(abs(q[:,active]-base['dof_pos'][:,active]).max()),source_sha256={str(x):sha(x) for x in [MOTION,OBJECT,URDF]},object_translation_change_m=stats(np.linalg.norm(op-np.column_stack([np.interp(np.arange(len(op))/base['fps'],np.arange(501)/120,np.load(OBJECT)['trans'][:,j]) for j in range(3)]),axis=1)),joint_checks=[dict(joint=names[j],maximum_speed_rad_s=float(speeds[:,j].max()),limit_rad_s=float(velocity[j]),violations=int((speeds[:,j]>velocity[j]+1e-6).sum())) for j in range(29)],elapsed_s=time.monotonic()-started,outputs={str(x):sha(x) for x in [path,path.with_suffix('.npz')]},no_policy_or_training=True)
    write_json(out/'contact_plan.json',plan);write_json(out/'solver_report.json',report);write_json(out/'solver_trace.json',trace);write_json(ROOT/'reports'/('solver_'+args.name+'.json'),report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
