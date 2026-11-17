"""Variant-specific comparison plus unchanged independent geometry gates."""
import sys,argparse,pickle,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import ROOT,DATA,VIBRATION_DATA,install,sha,write_json
install()
import numpy as np
from scipy.spatial.transform import Rotation
from validate_lift_hold import load_contract,array_gates,independent_mujoco_fk,collision_assessment
from lift_hold_validation_common import pose_samples,world_fk,contact_assessment,stats,rotation_delta
p=argparse.ArgumentParser();p.add_argument('--motion',type=Path,required=True);p.add_argument('--loader-samples',type=Path);p.add_argument('--report',type=Path,required=True);p.add_argument('--collision-rate',type=float,default=30);args=p.parse_args()
if VIBRATION_DATA not in args.motion.resolve().parents or ROOT not in args.report.resolve().parents:raise ValueError('Output boundary')
path,m,obj,u,j,n,b=load_contract(args.motion);source=DATA/'final/lift_hold_001_g1.pkl';old=pickle.load(source.open('rb'));config=json.loads((ROOT/'config/remove_vibration.json').read_text());limits=config['gates'];sourcehash=sha(source)
if sourcehash!=config['source_sha256']:raise ValueError('Accepted source changed')
array,diag=array_gates(m,obj,u,j,n,b);fk=independent_mujoco_fk(m,u,n,b);array['independent_mujoco_FK_10um']=fk['max_world_position_error_m']<1e-5;array['independent_mujoco_FK_rotation_1urad']=fk['max_world_rotation_error_rad']<1e-6
if args.loader_samples:
 with np.load(args.loader_samples,allow_pickle=False) as z:
  if str(z['motion_sha256'])!=sha(path):raise ValueError('Trace hash mismatch')
  t=z['times_s'];sample={k:z[k] for k in ['root_pos','root_rot','dof_pos']}
else:
 t=np.linspace(0,(len(m['root_pos'])-1)/m['fps'],7621);sample=pose_samples(m,t)
lower=np.array([float(x.find('limit').get('lower')) for x in j]);upper=np.array([float(x.find('limit').get('upper')) for x in j]);vmax=np.array([float(x.find('limit').get('velocity')) for x in j])
array['dense_URDF_position_limits']=bool(((sample['dof_pos']>=lower-1e-6)&(sample['dof_pos']<=upper+1e-6)).all())
array['dense_URDF_speed_limits']=bool((abs(np.diff(sample['dof_pos'],axis=0))/np.diff(t)[:,None]<=vmax+1e-6).all())
contact,cr,newfk,tr=contact_assessment(sample,u,n,b,t);baseline=pose_samples(old,t);oldfk=world_fk(baseline,u,n,b)
rootdev=np.linalg.norm(sample['root_pos']-baseline['root_pos'],axis=1);rootangle=np.degrees(rotation_delta(sample['root_rot'],baseline['root_rot']))
hand={}
for h in ['left_rubber_hand','right_rubber_hand']:
 pos=np.linalg.norm(newfk[3][:,b.index(h)]-oldfk[3][:,b.index(h)],axis=1)
 angle=np.degrees(Rotation.from_matrix(np.swapaxes(oldfk[2]@oldfk[1][h],1,2)@(newfk[2]@newfk[1][h])).magnitude())
 hand[h]=dict(position_m=stats(pos),angle_deg=stats(angle))
knots=np.arange(len(m['root_pos']))/m['fps'];a,c=config['hold_start_end_s'];outside=(knots<=a)|(knots>=c);core=(knots>=a+config['transition_duration_each_end_s'])&(knots<=c-config['transition_duration_each_end_s'])
gates=dict(object_bytes_exact=sha(path.with_suffix('.npz'))==sha(source.with_suffix('.npz')),
 duration_exact=len(m['root_pos'])==len(old['root_pos']) and m['fps']==old['fps'],
 outside_hold_exact=all(np.array_equal(m[k][outside],old[k][outside]) for k in ['root_pos','root_rot','dof_pos','local_body_pos']),
 static_hold_core_exact=all(np.array_equal(m[k][core],np.repeat(m[k][core][0:1],core.sum(),axis=0)) for k in ['root_pos','root_rot','dof_pos']),
 root_deviation_bounded=rootdev.max()<=limits['maximum_root_displacement_m'] and rootangle.max()<=limits['maximum_root_orientation_change_deg'],
 hand_deviation_bounded=all(x['position_m']['max']<=limits['maximum_hand_displacement_m'] and x['angle_deg']['max']<=limits['maximum_hand_orientation_change_deg'] for x in hand.values()))
gates={k:bool(v) for k,v in gates.items()}
metrics={}
for label,motion in [('before',old),('after',m)]:
 mask=(knots>=a+config['transition_duration_each_end_s'])&(knots<=c-config['transition_duration_each_end_s']);values=motion['root_pos'][mask];q=motion['dof_pos'][mask]
 metrics[label]=dict(root_speed_rms_m_s=float(np.sqrt(np.mean((np.diff(values,axis=0)*m['fps'])**2))),joint_speed_rms_rad_s=float(np.sqrt(np.mean((np.diff(q,axis=0)*m['fps'])**2))))
# Include both transition intervals and the entire object hold in derivative reports.
phase_metrics={}
for label,motion in [('before',old),('after',m)]:
 phase_metrics[label]={}
 for phase,mask in [('whole_hold',(knots>=a)&(knots<=c)),('entry',(knots>=a)&(knots<=a+.75)),('exit',(knots>=c-.75)&(knots<=c))]:
  phase_metrics[label][phase]={}
  for key in ['root_pos','dof_pos']:
   v=np.diff(motion[key][mask],axis=0)*m['fps'];acc=np.diff(v,axis=0)*m['fps']
   phase_metrics[label][phase][key]=dict(speed_rms=float(np.sqrt(np.mean(v*v))),speed_max=float(np.max(abs(v))),acceleration_rms=float(np.sqrt(np.mean(acc*acc))),acceleration_max=float(np.max(abs(acc))))
from scipy.signal import butter,sosfiltfilt
windows={}
high_frequency={}
for label,sm,fks in [('before',baseline,oldfk),('after',sample,newfk)]:
 channels={'root_position':sm['root_pos'],'joint_position':sm['dof_pos']}
 rootomega=(Rotation.from_quat(sm['root_rot'][:-1]).inv()*Rotation.from_quat(sm['root_rot'][1:])).as_rotvec()/np.diff(t)[:,None]
 windows[label]={}
 for h in ['left_rubber_hand','right_rubber_hand']:channels[h]=fks[3][:,b.index(h)]
 for boundary in [a,a+.75,c-.75,c]:
  mask=(t>=boundary-.1)&(t<=boundary+.1);indices=np.flatnonzero(mask);dt=np.diff(t[mask]);entry={}
  for name,values in channels.items():
   v=np.diff(values[mask],axis=0)/dt[:,None];acc=np.diff(v,axis=0)/((dt[1:]+dt[:-1])/2)[:,None]
   entry[name]=dict(speed_max=float(np.linalg.norm(v,axis=1).max()),acceleration_max=float(np.linalg.norm(acc,axis=1).max()))
  entry['root_angular_speed_max_rad_s']=float(np.linalg.norm(rootomega[indices[:-1]],axis=1).max());windows[label][str(boundary)]=entry
 high_frequency[label]={}
 sos=butter(4,3.,'highpass',fs=1/np.median(np.diff(t)),output='sos');hold=(t>=a)&(t<=c)
 for name,values in channels.items():
  residual=sosfiltfilt(sos,values,axis=0);high_frequency[label][name]=float(np.sqrt(np.mean(residual[hold]**2)))
report=dict(status='PENDING_COLLISIONS',motion=str(path),motion_sha256=sha(path),variant_config=config,array_gates=array,contact_gates=contact,comparison_gates=gates,diagnostics=diag,independent_fk=fk,contact=cr,comparison=dict(root_position_m=stats(rootdev),root_angle_deg=stats(rootangle),hands=hand),vibration_core=metrics,phase_derivatives=phase_metrics,boundary_actual600hz_windows=windows,high_frequency_rms_above_3hz=high_frequency,physical_grasp_verified=False,policy_executed=False,training_executed=False)
write_json(args.report,report)
cg,collision=collision_assessment(m,u,n,b,args.collision_rate,args.loader_samples,sha(path));report.update(collision_gates=cg,collision=collision)
report['accepted_source_preserved']=sha(source)==sourcehash
allg={**array,**contact,**gates,**cg,'accepted_source_preserved':report['accepted_source_preserved']};failed=[k for k,v in allg.items() if not v];report['failed_gates']=failed;report['status']='PASS_REMOVE_VIBRATION_REFERENCE' if not failed else 'FAIL_REMOVE_VIBRATION_REFERENCE'
write_json(args.report,report);print(json.dumps(dict(status=report['status'],failed=failed,comparison=report['comparison'],vibration=metrics)))
if failed:raise SystemExit(1)
