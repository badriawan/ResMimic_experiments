"""Solver-side collision diagnostic; independent validation remains separate."""
import argparse,sys,pickle,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,install,write_json
install()
import numpy as np
from geometry_oracle import Oracle
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--stride',type=int,default=20);args=p.parse_args()
path=ROOT/'candidates'/args.name/'interpose_floorlamp_g1_hand_contact.pkl';m=pickle.load(path.open('rb'));o=np.load(path.with_suffix('.npz'));g=Oracle();mins={};bad=[]
for frame in range(0,len(m['dof_pos']),args.stride):
 g.set_frame(m['dof_pos'][frame],m['root_pos'][frame],m['root_rot'][frame],o['trans'][frame],o['rot'][frame])
 for a,b in g.self_pairs+[(x,'floorlamp') for x in g.object_nonhand_links]+[(s+'_rubber_hand','floorlamp') for s in ['left','right']]:
  z=g.fast_pair_distance(a,b,threshold=.003) if b!='floorlamp' or 'hand' not in a else g.pair_distance(a,b)
  d=z['distance_m'];k=a+'|'+b
  if k not in mins or d<mins[k]['distance_m']:mins[k]=dict(frame=int(frame),distance_m=float(d),signed=z.get('signed_distance_valid'))
  if d<-.0002:bad.append(dict(frame=int(frame),pair=k,distance_m=float(d)))
print('worst',sorted(mins.items(),key=lambda kv:kv[1]['distance_m'])[:15]);print('badcount',len(bad))
write_json(ROOT/'reports'/('solver_coarse_'+args.name+'.json'),dict(minima=mins,bad=bad))
