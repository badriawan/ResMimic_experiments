"""Audit placement/settling tail and exterior joins of the validated v2 candidate."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import ROOT,install,sha,write_json
install()
import numpy as np
from scipy.signal import butter,sosfiltfilt
from validate_lift_hold import load_contract
from validate_lowering_vibration import derivative_metrics,angular_metrics
from lift_hold_validation_common import pose_samples,world_fk
validation=ROOT/'reports/lowering_v2_attempt04_actual600hz_validation.json';r=json.loads(validation.read_text());assert r['status']=='PASS_LOWERING_REFERENCE'
source,old,obj,u,j,n,b=load_contract(r['source']);candidate=Path(r['motion']);trace=Path(r['actual_loader_samples'])
assert sha(source)==r['source_sha256'] and sha(candidate)==r['motion_sha256']
with np.load(trace,allow_pickle=False) as z:
 assert str(z['motion_sha256'])==sha(candidate)
 times=z['times_s'].copy();new={k:z[k].copy() for k in ['root_pos','root_rot','dof_pos']};newworld=z['world_link_pos'].copy()
before=pose_samples(old,times);oldworld=world_fk(before,u,n,b)[3]
config=r['configuration'];start,end=config['edit_start_end_s'];lowerend=r['lowering_start_end_s'][1]
phases={'settling_tail':[lowerend,end],'entry_transition':[start,start+.6],'exit_transition':[end-.6,end],'exterior_join_100ms':[end-.1,end+.1]}
metrics={};sos=butter(4,3.,'highpass',fs=600,output='sos')
for label,sample,world in [('before',before,oldworld),('after',new,newworld)]:
 metrics[label]={};channels={'root_position':sample['root_pos'],'joint_position':sample['dof_pos']}
 for hand in ['left_rubber_hand','right_rubber_hand']:channels[hand]=world[:,b.index(hand)]
 for phase,(a,c) in phases.items():
  mask=(times>=a-1e-10)&(times<=c+1e-10);row={}
  for name,values in channels.items():
   row[name]=derivative_metrics(values[mask],times[mask]);hf=sosfiltfilt(sos,values,axis=0)
   row[name]['HF_component_RMS_above_3Hz']=float(np.sqrt(np.mean(hf[mask]**2)))
  row['root_angular']=angular_metrics(sample['root_rot'][mask],times[mask]);metrics[label][phase]=row
write_json(ROOT/'reports/lowering_v2_tail_audit.json',dict(status='AUDITED_METRICS_WITH_COMPLETE_GEOMETRY_PASS',phases_s=phases,metrics=metrics,
    geometry_validation=str(validation),geometry_validation_sha256=sha(validation),motion_sha256=sha(candidate),source_sha256=sha(source),trace_sha256=sha(trace),script_sha256=sha(Path(__file__)),
    interpretation='Tail10.7–11.5s is included in full600Hz60paircollision validation. Exact exterior source poses resume at11.5s; global C1/C2 continuity is not guaranteed. Derivatives include source variation, piecewise-linear interpolation knots and float32 loader phase effects. Metrics are descriptive, not force/contact certification.'))
print('Tail audit complete')
