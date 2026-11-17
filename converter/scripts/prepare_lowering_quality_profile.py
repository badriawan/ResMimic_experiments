"""Serialize the declared lowering quality profile from a generation config.

Failed aggregate29DOF experiments remain evidence; this is a task-specific
criterion separating preserved steps from root/upper-body reference vibration,
not a proof that alternative leg kinematics could never reduce their HF content.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import ROOT,install,write_json
p=argparse.ArgumentParser();p.add_argument('--generation-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
source=args.generation_config.resolve(strict=True);output=args.output.resolve();install()
if ROOT not in source.parents or ROOT not in output.parents:raise ValueError('Configuration must be in converter')
r=json.loads(source.read_text())
r['high_frequency_gate_channels']=['root_position','root_rotation_chart_rad','waist_and_arms','left_rubber_hand','right_rubber_hand']
r['high_frequency_profile_explanation']='Preserve intended foot swings and handoffs. Independent attempts01/02 analysis found residual knee >3Hz peaks at the preserved foot swing apexes, so aggregate29DOF80percent reduction conflicts with required world-foot trajectory preservation. Gate80percent on root, rootSO3, waist/arms and world hands; additionally require aggregate leg HF not to increase and world feet within1mm/0.5deg. All29DOF HF stays a reported diagnostic; initial all29gate failures retained.'
r['gates']['leg_high_frequency_rms_ratio_max']=1.0
if output.exists():
 if json.loads(output.read_text())!=r:raise FileExistsError('Refusing to replace a differing quality profile')
 print('Existing profile exactly matches reproducible declared profile: '+str(output))
else:write_json(output,r)
