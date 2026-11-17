"""Independent dense geometry validation of a coupled robot/object candidate.

All robot/source collision queries are static. PASS is not a force, PhysX
cooking, balance, approach, policy tracking, or physical grasp certificate.
"""
import argparse,json,pickle,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,ASSETS,URDF,SOURCE,MOTION,OBJECT,BASELINE,install,sha,write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--rate',type=float,default=2400);parser.add_argument('--tag',default=None);args=parser.parse_args();install()
    import numpy as np,trimesh
    from scipy.spatial.transform import Rotation,Slerp
    from geometry_oracle import Oracle
    sys.path.insert(0,str(REPO/'legged_gym/legged_gym/scripts'))
    from convert_gmr_qpos import forward_kinematics,foot_clearance
    path=args.candidate.resolve(strict=True)
    if ROOT not in path.parents and path!=BASELINE.resolve():raise ValueError('Candidate outside arm_smoothing')
    tag=args.tag or path.parent.name
    if not tag.replace('_','').replace('-','').isalnum():raise ValueError('Invalid tag')
    with path.open('rb') as f:m=pickle.load(f)
    with BASELINE.open('rb') as f:source=pickle.load(f)
    with BASELINE.open('rb') as f:baseline=pickle.load(f)
    baseline_plan=json.loads((BASELINE.parent/'contact_plan.json').read_text())
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:o={k:z[k].copy() for k in z.files}
    with np.load(BASELINE.with_suffix('.npz'),allow_pickle=False) as z:original_object={k:z[k].copy() for k in z.files}
    plan=json.loads((path.parent/'contact_plan.json').read_text());robot=ET.parse(URDF).getroot()
    joints=[j for j in robot.findall('joint') if j.get('type') in ('revolute','continuous')];names=[j.get('name') for j in joints]
    lo=np.asarray([float(j.find('limit').get('lower')) for j in joints]);hi=np.asarray([float(j.find('limit').get('upper')) for j in joints]);speedlimits=np.asarray([float(j.find('limit').get('velocity')) for j in joints])
    inactive=np.asarray([i for i,name in enumerate(names) if not ('shoulder' in name or 'elbow' in name or 'wrist' in name)])
    n=len(m['root_pos']);fps=float(m['fps']);duration=(n-1)/fps;native_t=np.arange(n)/fps
    source_fps=float(source['fps']);source_t=np.arange(len(source['root_pos']))/source_fps
    clockfile=path.parent/'transition_clock.npz'
    with np.load(clockfile,allow_pickle=False) as z:clock={k:z[k].copy() for k in z.files}
    transition=plan['transition'];hold=float(transition['hold_s']);ramp=float(transition['ramp_s']);start=float(transition['source_start_time_s'])
    tau=clock['source_reference_time_s'];join_out=int(round((hold+ramp)*fps));join_source=int(round((start+ramp/2)*source_fps));hold_end=int(round(hold*fps))
    def analytic_clock(query):
        u=np.clip((query-hold)/ramp,0,1)
        phase=start+ramp*(2.5*u**4-3*u**5+u**6)+np.maximum(query-hold-ramp,0)
        speed=np.where(query>=hold+ramp,1,10*u**3-15*u**4+6*u**5)
        acceleration=(30*u**2-60*u**3+30*u**4)/ramp
        return phase,speed,acceleration
    expected_tau,expected_rate,expected_acc=analytic_clock(native_t)
    source_interp=lambda array,query:np.stack([np.interp(query,source_t,array[:,k]) for k in range(array.shape[1])],axis=1)
    quat_error=lambda a,b:float(np.max(np.minimum(np.linalg.norm(a-b,axis=1),np.linalg.norm(a+b,axis=1))))
    invariant_errors=dict(root_position_m=float(np.max(abs(m['root_pos']-source_interp(source['root_pos'],tau)))),all_joint_rad=float(np.max(abs(m['dof_pos']-source_interp(source['dof_pos'],tau)))),root_quaternion_component=quat_error(m['root_rot'],Slerp(source_t,Rotation.from_quat(source['root_rot']))(tau).as_quat()),object_position_m=float(np.max(abs(o['trans']-source_interp(original_object['trans'],tau)))),object_quaternion_component=quat_error(o['rot'],Slerp(source_t,Rotation.from_quat(original_object['rot']))(tau).as_quat()),capture_clock_s=float(np.max(abs(o['source_time_s']-np.interp(tau,source_t,original_object['source_time_s'])))))
    hold_arrays=[m[k] for k in ['root_pos','root_rot','dof_pos','local_body_pos']]+[o[k] for k in ['trans','rot','source_time_s']]
    checks=dict(frames_and_fps=bool(fps==source_fps==480 and abs(duration-(source_t[-1]-start+hold+ramp/2))<2e-12),finite=all(np.isfinite(v).all() for v in m.values() if isinstance(v,np.ndarray)) and all(np.isfinite(v).all() for v in o.values()),object_shapes=o['trans'].shape==(n,3) and o['rot'].shape==(n,4),unit_quaternions=np.max(abs(np.linalg.norm(m['root_rot'],axis=1)-1))<1e-6 and np.max(abs(np.linalg.norm(o['rot'],axis=1)-1))<1e-6,joint_position_limits=np.all((m['dof_pos']>=lo-1e-8)&(m['dof_pos']<=hi+1e-8)),joint_velocity_limits=np.max(np.abs(np.diff(m['dof_pos'],axis=0))*fps/speedlimits)<=1+1e-6,shared_source_path_native=all(value<2e-12 for value in invariant_errors.values()),independent_clock_formula=np.max(abs(tau-expected_tau))<2e-12 and np.max(abs(clock['clock_rate']-expected_rate))<2e-12 and np.max(abs(clock['clock_acceleration_s_inv']-expected_acc))<2e-12,clock_monotone_no_overspeed=np.all(np.diff(tau)>=-1e-12) and np.max(np.diff(tau)*fps)<=1+1e-10 and np.min(clock['clock_rate'])>=0 and np.max(clock['clock_rate'])<=1,clock_hold_and_suffix_C2=bool(np.allclose(analytic_clock(np.array([hold,hold+ramp]))[1],[0,1],atol=1e-12) and np.max(abs(analytic_clock(np.array([hold,hold+ramp]))[2]))<1e-12),stationary_prefix_exact=all(np.array_equal(a[:hold_end+1],np.repeat(a[:1],hold_end+1,axis=0)) for a in hold_arrays),initial_finite_difference_velocities_zero=all(np.max(abs(np.diff(a[:hold_end+1],axis=0)))==0 for a in hold_arrays),source_suffix_exact=all(np.array_equal(m[k][join_out:],source[k][join_source:]) for k in ['root_pos','root_rot','dof_pos','local_body_pos']) and all(np.array_equal(o[k][join_out:],original_object[k][join_source:]) for k in ['trans','rot','source_time_s']),source_suffix_clock_1x=np.max(abs(np.diff(tau[join_out:])*fps-1))<1e-10,source_final_pose_and_clock_preserved=tau[-1]==source_t[-1] and all(np.array_equal(m[k][-1],source[k][-1]) for k in ['root_pos','root_rot','dof_pos']) and all(np.array_equal(o[k][-1],original_object[k][-1]) for k in ['trans','rot','source_time_s']))
    contact_keys=['pad_local','pad_normal_local','pad_triangle_local','target_point_object','outward_normal_object','target_gap_m','tangent_local']
    checks['material_contact_geometry_unchanged']=all(np.array_equal(np.asarray(plan['hands'][side][key]),np.asarray(baseline_plan['hands'][side][key])) for side in ['left','right'] for key in contact_keys)
    local,localrot=forward_kinematics(robot,m['dof_pos'],names,m['link_body_list']);checks['stored_local_body_pos_matches_independent_FK']=np.max(abs(local-m['local_body_pos']))<2e-7
    # Candidate intended interpolation is linear joint/position and shortest-path SLERP.
    count=int(round(duration*args.rate))+1;t=np.linspace(0,duration,count)
    interp=lambda array:np.stack([np.interp(t,native_t,array[:,k]) for k in range(array.shape[1])],axis=1)
    q=interp(m['dof_pos']);rp=interp(m['root_pos']);rq=Slerp(native_t,Rotation.from_quat(m['root_rot']))(t).as_quat();op=interp(o['trans']);oq=Slerp(native_t,Rotation.from_quat(o['rot']))(t).as_quat()
    rr=Rotation.from_quat(rq).as_matrix();ro=Rotation.from_quat(oq).as_matrix();lp,lr=forward_kinematics(robot,q,names,m['link_body_list'])
    body_names=list(m['link_body_list']);worldpos=rp[:,None,:]+np.einsum('tij,tkj->tki',rr,lp)
    worldrot={name:rr@lr[name] for name in body_names}
    feet=foot_clearance(robot,lp,lr,body_names,rp,rr)
    # Native knots preserve the baseline path exactly; interpolation between the
    # newly warped knots is a separate, explicitly measured sampling error.
    dense_tau=analytic_clock(t)[0]
    dense_errors=dict(root_position_m=float(np.max(abs(rp-source_interp(source['root_pos'],dense_tau)))),joint_position_rad=float(np.max(abs(q-source_interp(source['dof_pos'],dense_tau)))),object_position_m=float(np.max(abs(op-source_interp(original_object['trans'],dense_tau)))))
    g=Oracle();pairnames=[a+'|'+b for a,b in g.self_pairs]
    selfdist=np.zeros((count,len(g.self_pairs)));selfvalid=np.ones_like(selfdist,dtype=bool)
    objectdist=np.zeros((count,len(g.object_nonhand_links)));objectvalid=np.ones_like(objectdist,dtype=bool)
    handdist=np.zeros((count,2));handvalid=np.zeros((count,2),bool);paderr=np.zeros((count,2));normalerr=np.zeros((count,2));surfacegap=np.zeros((count,2));plane_nonpenetration=np.zeros((count,2));floor=np.zeros(count)
    patch_checks={};stem=trimesh.load_mesh(ASSETS/'floorlamp_stem_collision.obj',process=True);stemhull=stem.convex_hull;equations=__import__('scipy.spatial',fromlist=['ConvexHull']).ConvexHull(np.asarray(stem.vertices)).equations
    for side in ['left','right']:
        h=plan['hands'][side];mesh=trimesh.load_mesh(URDF.parent/'meshes'/(h['link']+'.STL'),process=True);vertices=np.asarray(mesh.vertices)
        pad=np.asarray(h['pad_local']);normal=np.asarray(h['pad_normal_local']);normal/=np.linalg.norm(normal)
        inward_violation=float(np.max((vertices-pad)@normal));triangle=np.asarray(h['pad_triangle_local']);anchor=np.asarray(h['target_point_object'])
        hullmax=float(np.max(equations[:,:3]@anchor+equations[:,3]));normalo=np.asarray(h['outward_normal_object']);facet=np.max(np.asarray(stem.vertices)@normalo)
        patch_checks[side]=dict(pad_on_actual_hand_support_hull=abs(inward_violation)<2e-6,triangle_centroid_matches_pad=np.max(abs(triangle.mean(axis=0)-pad))<2e-6,triangle_vertices_match_actual_mesh=bool(np.min(np.linalg.norm(triangle[:,None,:]-vertices[None,:,:],axis=2),axis=1).max()<2e-7),triangle_finite_area=bool(np.linalg.norm(np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]))/2>1e-6),object_anchor_on_stem_surface=abs(hullmax)<2e-6,object_support_facet=abs(anchor@normalo-facet)<2e-6,pad_support_plane_max_violation_m=inward_violation,object_anchor_hull_max_m=hullmax)
    patch_checks={side:{key:bool(value) if isinstance(value,np.bool_) else value for key,value in item.items()} for side,item in patch_checks.items()}
    checks['contact_patch_source_geometry']=all(all(v for k,v in item.items() if isinstance(v,(bool,np.bool_))) for item in patch_checks.values())
    outward=np.asarray([plan['hands'][side]['outward_normal_object'] for side in ['left','right']]);checks['opposed_object_normals']=float(outward[0]@outward[1])<-.99
    started=time.monotonic()
    for frame in range(count):
        poses={name:worldpos[frame,body_names.index(name)] for name in body_names};poses['floorlamp']=op[frame]
        rotations={name:worldrot[name][frame] for name in body_names};rotations['floorlamp']=ro[frame]
        g._bounds_cache={};g.set_link_poses(np.asarray([poses[name] for name in g.links]),np.asarray([rotations[name] for name in g.links]));g.objpos=op[frame];g.objrot=ro[frame];floor[frame]=g.floor_gap()
        for index,(a,b) in enumerate(g.self_pairs):
            value=g.fast_pair_distance(a,b,threshold=.003);selfdist[frame,index]=value['distance_m'];selfvalid[frame,index]=value['signed_distance_valid']
        for index,a in enumerate(g.object_nonhand_links):
            value=g.fast_pair_distance(a,'floorlamp',threshold=.003);objectdist[frame,index]=value['distance_m'];objectvalid[frame,index]=value['signed_distance_valid']
        for index,side in enumerate(['left','right']):
            h=plan['hands'][side];name=h['link'];pad=poses[name]+rotations[name]@np.asarray(h['pad_local']);target=op[frame]+ro[frame]@(np.asarray(h['target_point_object'])+float(h.get('target_gap_m',0))*np.asarray(h['outward_normal_object']));normal=rotations[name]@np.asarray(h['pad_normal_local']);out=ro[frame]@np.asarray(h['outward_normal_object']);paderr[frame,index]=np.linalg.norm(pad-target);normalerr[frame,index]=np.degrees(np.arccos(np.clip(normal@(-out),-1,1)))
            value=g.hand_lamp_distance(name,out);handdist[frame,index]=value['distance_m'];handvalid[frame,index]=value['signed_distance_valid']
            # Support-plane certificate is independent of the near-zero CCD result.
            support_hand=max(g._support_point(gid,-out)@(-out) for gid in g.geom_ids[name]['capsule'])
            support_object=float(g._support_point(g.geom_ids['floorlamp']['capsule'][1],out)@out)
            plane_nonpenetration[frame,index]=-support_hand-support_object
            anchor=target-float(h.get('target_gap_m',0))*out;surfacegap[frame,index]=(pad-anchor)@out
        if frame%500==0:print('Independent frame%d/%d elapsed%.1fs'%(frame,count,time.monotonic()-started),flush=True)
    # Require actual support-plane nonpenetration plus a close actual pad anchor.
    # Other hand/lamp components can be closer; depth queries are still retained.
    checks.update(dense_contact_point_error_2mm=float(paderr.max())<=.002,dense_contact_normal_error_5deg=float(normalerr.max())<=5,dense_hand_object_nonpenetration=bool((handdist>=-1e-6).all() and (plane_nonpenetration>=-1e-6).all() and handvalid.all()),dense_contact_gap_2mm=bool((surfacegap>=-1e-6).all() and (surfacegap<=.002).all()),dense_self_clearance_1mm=bool((selfdist>=.001).all() and selfvalid.all()),dense_lamp_nonhand_clearance_1mm=bool((objectdist>=.001).all() and objectvalid.all()),dense_object_floor_nonpenetration=bool(floor.min()>=-1e-6),dense_foot_floor_nonpenetration=bool(min(np.min(item['capsule_surrogate']) for item in feet.values())>=-1e-6))
    material_twist_deg=np.zeros((count,2))
    for index,side in enumerate(['left','right']):
        h=plan['hands'][side];out=np.asarray(h['outward_normal_object'],float);out/=np.linalg.norm(out)
        tangent_world=np.einsum('tij,j->ti',worldrot[h['link']],np.asarray(h['tangent_local']))
        tangent_object=np.einsum('tji,tj->ti',ro,tangent_world)
        tangent_object-=np.einsum('ti,i->t',tangent_object,out)[:,None]*out
        tangent_object/=np.linalg.norm(tangent_object,axis=1)[:,None]
        initial=tangent_object[0]
        angle=np.arctan2(np.einsum('ti,i->t',np.cross(initial,tangent_object),out),tangent_object@initial)
        material_twist_deg[:,index]=np.degrees(np.unwrap(angle))
    # Outside explicit contact phase, do not silently accept an incorrect phase plan.
    checks['declared_pregrasped_full_clip']=bool(plan.get('contact_from_start') and plan.get('contact_start_frame')==0 and plan.get('contact_end_frame')==n-1)
    pair_min=np.min(selfdist,axis=0);opmin=objectdist.min(axis=0)
    badself=[dict(pair=pairnames[i],minimum_distance_m=float(pair_min[i]),time_s=float(t[np.argmin(selfdist[:,i])]),signed_all_valid=bool(selfvalid[:,i].all())) for i in range(len(pairnames)) if pair_min[i]<.001 or not selfvalid[:,i].all()]
    badobject=[dict(link=name,minimum_distance_m=float(opmin[i]),time_s=float(t[np.argmin(objectdist[:,i])]),signed_all_valid=bool(objectvalid[:,i].all())) for i,name in enumerate(g.object_nonhand_links) if opmin[i]<.001 or not objectvalid[:,i].all()]
    np.savez_compressed(ROOT/('reports/validation_'+tag+'_dense_metrics.npz'),time_s=t,self_distances_m=selfdist,self_valid=selfvalid,object_nonhand_distances_m=objectdist,object_nonhand_valid=objectvalid,hand_object_distances_m=handdist,hand_object_valid=handvalid,pad_error_m=paderr,normal_error_deg=normalerr,object_floor_gap_m=floor,support_plane_gap_m=plane_nonpenetration,pad_surface_gap_m=surfacegap,material_twist_relative_initial_deg=material_twist_deg)
    report=dict(status='PASS_STATIC_REFERENCE_GEOMETRY' if all(checks.values()) else 'FAIL_STATIC_REFERENCE_GEOMETRY',checks={k:bool(v) for k,v in checks.items()},candidate=str(path),frames=n,fps=fps,dense_samples=count,dense_rate_hz=args.rate,scope='Static source convex hull/capsule geometry, linear joint/position interpolation and SLERP, no dynamics or grasp certificate',metrics=dict(max_pad_error_m=paderr.max(axis=0).tolist(),max_normal_error_deg=normalerr.max(axis=0).tolist(),minimum_hand_object_distance_m=handdist.min(axis=0).tolist(),minimum_support_plane_gap_m=plane_nonpenetration.min(axis=0).tolist(),pad_surface_gap_range_m=[surfacegap.min(axis=0).tolist(),surfacegap.max(axis=0).tolist()],minimum_self_clearance_m=float(selfdist.min()),minimum_lamp_nonhand_clearance_m=float(objectdist.min()),minimum_object_floor_gap_m=float(floor.min()),maximum_object_floor_gap_m=float(floor.max()),maximum_joint_velocity_ratio=float(np.max(np.abs(np.diff(m['dof_pos'],axis=0))*fps/speedlimits)),maximum_material_tangent_drift_deg=np.max(np.abs(material_twist_deg),axis=0).tolist()),joint_velocity_checks=[dict(name=name,maximum_rad_s=float(np.max(np.abs(np.diff(m['dof_pos'][:,j])))*fps),limit_rad_s=float(speedlimits[j]),ratio=float(np.max(np.abs(np.diff(m['dof_pos'][:,j])))*fps/speedlimits[j])) for j,name in enumerate(names)],conservation_proof=dict(source_fps=source_fps,candidate_fps=fps,tolerance_float64=2e-12,errors=invariant_errors,dense_interpolation_errors=dense_errors,meaning='All native poses follow one independently reconstructed source clock; source-native 1x suffix is exact. Dense output interpolation differs slightly from analytic clock sampling. C2 applies to the source clock, not the original piecewise-linear joint/position path.',clock_file=str(clockfile),source_start_time_s=start,hold_s=hold,ramp_s=ramp,source_join_frame=join_source,output_join_frame=join_out),patch_geometry=patch_checks,self_pair_names=pairnames,object_nonhand_links=g.object_nonhand_links,bad_self_pairs=badself,bad_object_nonhand=badobject,wall_time_s=time.monotonic()-started,input_sha256={str(p):sha(p) for p in [path,path.with_suffix('.npz'),path.parent/'contact_plan.json',BASELINE,BASELINE.with_suffix('.npz'),MOTION,OBJECT,URDF,ASSETS/'asset_profile.json']},limitations=['No actuator forces, friction equilibrium, balance, PhysX cooked shape, approach or free-object lifting verified.','Source convex hull distances may differ from simulator cooked collision shapes.','Declared geometric target gaps are not actual compressive forces.','C2 clock joins do not remove derivative discontinuities in the original source pose interpolant.','Two hard point contacts alone do not guarantee full wrench closure.'])
    write_json(ROOT/('reports/validation_'+tag+'_validation.json'),report);print(json.dumps({k:v for k,v in report.items() if k not in ['self_pair_names','input_sha256']},indent=2))
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
