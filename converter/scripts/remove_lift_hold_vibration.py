"""Make a separate static-core hold reference; no dynamics or policy execution."""
import sys,pickle,shutil,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import ROOT,REPO,DATA,VIBRATION_DATA,install,sha,write_json
install()
import numpy as np
from scipy.spatial.transform import Rotation
from correct_lift_hold import Chain,solve,quintic
from validate_lift_hold import load_contract
from lift_hold_validation_common import world_fk,stats

parser=argparse.ArgumentParser();parser.add_argument('--attempt',default='static_hold03');args=parser.parse_args()
source=DATA/'final/lift_hold_001_g1.pkl'
_,old,obj,urdf,joints,names,bodies=load_contract(source)
fps=float(old['fps']);t=np.arange(len(old['root_pos']))/fps
# Detected from the unchanged object plateau, not a guessed human phase.
z=obj['trans'][:,2];plateau=np.flatnonzero(abs(z-z.max())<1e-7)
a,b=int(plateau[0]),int(plateau[-1]);ramp=.75
weight=quintic((t-t[a])/ramp)*quintic((t[b]-t)/ramp)
anchor=(a+b)//2
config=dict(method='Static middle hold with fixed-endpoint quintic transitions and world-foot IK',
    source=str(source),source_sha256=sha(source),hold_start_end_s=[float(t[a]),float(t[b])],
    transition_duration_each_end_s=ramp,anchor_frame=anchor,anchor_time_s=float(t[anchor]),
    new_variant_comparison='Accepted final reference; original 10mm hand correction gate is not applicable to this requested motion edit.',
    gates=dict(object_bytes_exact=True,outside_hold_exact=True,core_pose_static=True,
        maximum_root_displacement_m=.10,maximum_root_orientation_change_deg=10.,
        maximum_hand_displacement_m=.10,maximum_hand_orientation_change_deg=35.,
        foot_penetration_m=.002,support_band_m=.005,near_floor_probe_speed_max_m_s=.1,
        near_floor_probe_speed_p95_m_s=.05,collision_minimum_gap_m=-.001),
    scope='Kinematic reference editing, not recovered source contact or physical grasp')
write_json(ROOT/'config/remove_vibration.json',config)
out=VIBRATION_DATA/'candidates'/args.attempt
out.mkdir(parents=True,exist_ok=False)
new={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in old.items()}
w=weight[:,None]
for key in ['root_pos','dof_pos']:
    boundary=np.where((t<=t[anchor])[:,None],old[key][a],old[key][b])
    new[key]=(1-w)*boundary+w*old[key][anchor]
r=Rotation.from_quat(np.where((t<=t[anchor])[:,None],old['root_rot'][a],old['root_rot'][b]));delta=(r.inv()*Rotation.from_quat(old['root_rot'][anchor])).as_rotvec()
new['root_rot']=(r*Rotation.from_rotvec(weight[:,None]*delta)).as_quat()
active=np.flatnonzero(weight>0)
_,lr,rr,wp=world_fk(old,urdf,names,bodies)
newrr=Rotation.from_quat(new['root_rot']).as_matrix()
lower=np.array([float(j.find('limit').get('lower')) for j in joints]);upper=np.array([float(j.find('limit').get('upper')) for j in joints])
traces={}
for foot in ['left_ankle_roll_link','right_ankle_roll_link']:
    chain=Chain(urdf,names,foot);idx=np.array(chain.indices);trace=[]
    # Preserve the already corrected world support anchor throughout this hold.
    for k in active:
        q=new['dof_pos'][k].copy()
        values,pe,re,calls=solve(chain,q,q[idx],new['root_pos'][k],newrr[k],wp[anchor,bodies.index(foot)],rr[anchor]@lr[foot][anchor],lower,upper,idx,max_nfev=100)
        new['dof_pos'][k,idx]=values;trace.append([pe,re])
    traces[foot]=dict(position_m=stats(np.array(trace)[:,0]),rotation_rad=stats(np.array(trace)[:,1]))
# Exact equality outside the requested interval, including its boundary knots.
for key in ['root_pos','root_rot','dof_pos']:
    new[key][weight==0]=old[key][weight==0]
# Use the exact same solved pose in the fully static core, removing solver noise.
core=weight>=1-1e-14
for key in ['root_pos','root_rot','dof_pos']:
    new[key][core]=new[key][anchor]
new['local_body_pos']=world_fk(new,urdf,names,bodies)[0]
path=out/'lift_hold_001_g1.pkl'
with path.open('xb') as f:pickle.dump(new,f,protocol=4)
shutil.copyfile(source.with_suffix('.npz'),path.with_suffix('.npz'))
write_json(ROOT/'reports'/('remove_vibration_generation_'+args.attempt+'.json'),dict(config=config,output=str(path),output_sha256=sha(path),leg_ik=traces,static_core_frames=int(core.sum()),static_core_start_end_s=t[np.flatnonzero(core)[[0,-1]]].tolist(),source_preserved=sha(source)==config['source_sha256']))
print(path)
