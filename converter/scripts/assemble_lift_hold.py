"""Combine validated leg and upper-body paths without reoptimizing either."""
import argparse
from pathlib import Path
import pickle
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import ROOT,REPO,DATA,install,sha,write_json
install()
import numpy as np
from lift_hold_validation_common import pose_samples,rotation_delta
sys.path.insert(0,str(REPO/'legged_gym/legged_gym/scripts'))
from convert_gmr_qpos import forward_kinematics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legs',type=Path,required=True)
    parser.add_argument('--upper',type=Path,required=True)
    parser.add_argument('--name',default='combined_attempt01')
    args=parser.parse_args()
    if not args.name.replace('_','').isalnum():raise ValueError('Unsafe candidate name')
    paths=[args.legs.resolve(strict=True),args.upper.resolve(strict=True)]
    if any(DATA not in p.parents for p in paths):raise ValueError('Inputs outside converted data root')
    motions=[]
    for p in paths:
        with p.open('rb') as f:motions.append(pickle.load(f))
    leg,upper=motions
    times=np.arange(len(leg['root_pos']))/float(leg['fps'])
    duration=(len(upper['root_pos'])-1)/float(upper['fps'])
    assert abs(times[-1]-duration)<1e-10
    sampled=pose_samples(upper,times)
    checks=dict(root_position_path_matches=bool(np.max(abs(leg['root_pos']-sampled['root_pos']))<1e-12),
                root_orientation_path_matches=bool(rotation_delta(leg['root_rot'],sampled['root_rot']).max()<1e-10),
                waist_path_matches=bool(np.max(abs(leg['dof_pos'][:,12:15]-sampled['dof_pos'][:,12:15]))<1e-12),
                body_order_matches=list(leg['link_body_list'])==list(upper['link_body_list']))
    if not all(checks.values()):raise ValueError('Cannot splice incompatible root/waist paths: '+str(checks))
    result={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in leg.items()}
    result['dof_pos'][:,15:]=sampled['dof_pos'][:,15:]
    urdf=ET.parse(REPO/'assets/g1/g1_custom_collision_29dof.urdf').getroot()
    names=[j.get('name') for j in urdf.findall('joint') if j.get('type')!='fixed']
    result['local_body_pos'],_=forward_kinematics(urdf,result['dof_pos'],names,result['link_body_list'])
    out=DATA/'candidates'/args.name
    out.mkdir(parents=True,exist_ok=False)
    path=out/'lift_hold_001_g1.pkl'
    with path.open('xb') as f:pickle.dump(result,f,protocol=4)
    with path.with_suffix('.npz').open('xb') as f:f.write(paths[0].with_suffix('.npz').read_bytes())
    checks.update(leg_joint_path_exact=bool(np.array_equal(result['dof_pos'][:,:12],leg['dof_pos'][:,:12])),
                  upper_joint_continuous_path_exact=bool(np.array_equal(result['dof_pos'][:,15:],sampled['dof_pos'][:,15:])),
                  root_array_preserved=bool(np.array_equal(result['root_pos'],leg['root_pos']) and np.array_equal(result['root_rot'],leg['root_rot'])),
                  object_bytes_exact=sha(path.with_suffix('.npz'))==sha(paths[0].with_suffix('.npz')))
    write_json(ROOT/'reports'/(args.name+'_assembly.json'),dict(status='PATH_ASSEMBLY_PASS_PENDING_INDEPENDENT_VALIDATION',
        motion=str(path),frames=len(times),fps=float(result['fps']),duration_s=float(times[-1]),checks=checks,
        legs_source=str(paths[0]),legs_sha256=sha(paths[0]),upper_source=str(paths[1]),upper_sha256=sha(paths[1]),
        output_sha256=sha(path),object_sha256=sha(path.with_suffix('.npz')),
        meaning='New legs with the exact continuous upper path of hand correction; no time change or reoptimization',
        policy_executed=False,training_executed=False))
    print(str(path))


if __name__=='__main__':main()
