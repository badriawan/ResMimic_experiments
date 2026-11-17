"""Measure free hand tangential twist and record explicit reproducibility metadata."""
import argparse,json,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,URDF,install,write_json,sha
install()
import numpy as np
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
sys.path.insert(0,str(REPO/'legged_gym/legged_gym/scripts'))
from convert_gmr_qpos import forward_kinematics
p=argparse.ArgumentParser();p.add_argument('name');args=p.parse_args();out=ROOT/'candidates'/args.name
m=pickle.load((out/'interpose_floorlamp_g1_hand_contact.pkl').open('rb'));o=np.load(out/'interpose_floorlamp_g1_hand_contact.npz');plan=json.loads((out/'contact_plan.json').read_text());trace=json.loads((out/'solver_trace.json').read_text());report=json.loads((out/'solver_report.json').read_text())
u=ET.parse(URDF).getroot();names=[j.get('name') for j in u.findall('joint') if j.get('type')!='fixed'];lp,rotations=forward_kinematics(u,m['dof_pos'],names,m['link_body_list']);rr=Rotation.from_quat(m['root_rot']).as_matrix();ro=Rotation.from_quat(o['rot']).as_matrix();measure={}
for side in ['left','right']:
 h=plan['hands'][side];handr=rr@rotations[h['link']];actual=np.einsum('tij,j->ti',handr,np.asarray(h['tangent_local']));target=np.einsum('tij,j->ti',ro,np.asarray(h['target_tangent_object']));error=np.degrees(np.arccos(np.clip(np.einsum('ti,ti->t',actual,target),-1,1)))
 object_tangent=np.einsum('tji,tj->ti',ro,actual);initial=object_tangent[0];drift=np.degrees(np.arccos(np.clip(object_tangent@initial,-1,1)))
 measure[side]=dict(bias_angle_degrees_min=float(error.min()),bias_angle_degrees_max=float(error.max()),change_from_initial_degrees_max=float(drift.max()))
 for row in trace:
  if row['side']==side:row['tangent_error_deg']=float(error[row['frame']])
all_angles=np.asarray([row['tangent_error_deg'] for row in trace]);report['tangent_error_deg']=dict(min=float(all_angles.min()),median=float(np.median(all_angles)),p95=float(np.percentile(all_angles,95)),max=float(all_angles.max()));report['free_tangent_twist']=measure
report['tangent_interpretation']='Measured angle to optional forward bias; unconstrained tangential rotation is permitted, and angle is not an acceptance gate. This is not certified finite-patch sticking.'
plan['tangent_interpretation']=report['tangent_interpretation'];plan['contact_model']='Fixed material contact center and opposing support normal (5DOF); free rotation about normal. No certified finite-patch no-slip or physical force closure.'
if args.name=='attempt08':
 report['solver_arguments']=dict(name='attempt08',forward=.38,lateral=0.,grip_height=.16,stagger=0.,sigma=10.,gap=.0005,coupled_seed=str(ROOT/'candidates/attempt07/interpose_floorlamp_g1_hand_contact.pkl'))
plan['solver_arguments']=report.get('solver_arguments',plan.get('solver_arguments'))
report['solver_implementation_sha256']={str(p):sha(p) for p in (ROOT/'scripts').glob('solver_*.py')};report['entrypoint_sha256_current']=sha(ROOT/'scripts/solve_contact_reference.py');report['metadata_note']='Free-tangent metric recomputed from final FK after solve; state files unchanged.'
if args.name=='attempt08':report['metadata_note']+=' Current entrypoint adds metadata-only logging relative to loaded solver run.'
else:
    options=report.get('solver_arguments',{})
    plan['tangent_regularization']=dict(weight=options.get('tangent_weight',0),target='Initial object-frame tangent' if options.get('lock_initial_tangent') else 'Forward tangent',hard_constraint=False)
    if options.get('lock_initial_tangent'):plan['strategy']=plan['strategy'].replace('weak forward bias','soft initial object-frame tangential orientation regularizer')
points=np.stack([o['trans']+np.einsum('tij,j->ti',ro,np.asarray(plan['hands'][side]['target_point_object'])+np.asarray(plan['hands'][side]['outward_normal_object'])*plan['hands'][side]['target_gap_m']) for side in ['left','right']],axis=1)
normals=np.stack([-np.einsum('tij,j->ti',ro,np.asarray(plan['hands'][side]['outward_normal_object'])) for side in ['left','right']],axis=1)
tangents=np.stack([np.einsum('tij,j->ti',ro,np.asarray(plan['hands'][side]['target_tangent_object'])) for side in ['left','right']],axis=1)
np.savez_compressed(out/'solver_targets.npz',point_world=points,normal_world=normals,tangent_world=tangents)
report['metadata_note']+=' Final diagnostic target arrays recomputed from exported plan/object; no trajectory state changed.'
write_json(out/'contact_plan.json',plan);write_json(out/'solver_report.json',report);write_json(out/'solver_trace.json',trace);write_json(ROOT/'reports'/('solver_'+args.name+'.json'),report)
print(json.dumps(dict(status=report['status'],free_tangent_twist=measure,gates=report['gates']),indent=2))
