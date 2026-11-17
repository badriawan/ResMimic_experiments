"""Publish the validated immutable motion pair and its provenance."""
import sys,shutil,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import ROOT,DATA,VIBRATION_DATA,install,sha,write_json
install()
reference=json.loads((ROOT/'reports/remove_vibration_actual600hz_final.json').read_text())
loader=json.loads((ROOT/'reports/remove_vibration_loader_final.json').read_text())
video=json.loads((ROOT/'reports/remove_vibration_video_final.json').read_text())
assert reference['status']=='PASS_REMOVE_VIBRATION_REFERENCE' and not reference['failed_gates']
assert loader['status']=='PASS_ACTUAL_LOADER' and loader['quality_status']=='PASS_SAMPLED_FOOT_CONTACT'
assert video['status']=='PASS_REFERENCE_VIDEO' and all(video['checks'].values())
source=Path(reference['motion']);assert sha(source)==reference['motion_sha256']==loader['motion_sha256']
assert video['source_files'][str(source)]==sha(source)
published={}
for suffix in ['.pkl','.npz']:
 dst=VIBRATION_DATA/('lift_hold_001_g1'+suffix)
 if dst.exists():raise FileExistsError(dst)
 shutil.copyfile(source.with_suffix(suffix),dst)
 assert sha(dst)==sha(source.with_suffix(suffix))
 published[str(dst)]=sha(dst)
assets=VIBRATION_DATA/'assets';assets.mkdir(exist_ok=True)
shutil.copyfile(DATA/'assets/floorlamp.obj',assets/'floorlamp.obj');published[str(assets/'floorlamp.obj')]=sha(assets/'floorlamp.obj')
scripts=ROOT/'scripts'
manifest=dict(status='PASS_KINEMATIC_REFERENCE',published=published,candidate=str(source),
 accepted_input={str(p):sha(p) for p in [DATA/'final/lift_hold_001_g1.pkl',DATA/'final/lift_hold_001_g1.npz']},
 validation_reports={str(p):sha(p) for p in [ROOT/'reports/remove_vibration_actual600hz_final.json',ROOT/'reports/remove_vibration_loader_final.json',ROOT/'reports/remove_vibration_video_final.json']},
 implementation_sha256={str(p):sha(p) for p in [scripts/x for x in ['remove_lift_hold_vibration.py','validate_remove_vibration.py','validate_lift_hold.py','validate_lift_hold_loader.py','lift_hold_validation_common.py','lift_hold_collision_oracle.py','render_lift_hold_reference.py','converter_common.py']]},
 video=str(ROOT/'videos/lift_hold_001_remove_vibration_final.mp4'),video_sha256=sha(ROOT/'videos/lift_hold_001_remove_vibration_final.mp4'),
 configuration=str(ROOT/'config/remove_vibration.json'),configuration_sha256=sha(ROOT/'config/remove_vibration.json'),
 note='Motion pair copied byte-for-byte from the exact validated and rendered candidate; reference kinematics only. Source object retained including known nominal floor penetration. Hold transitions follow fixed-endpoint quintic curves.',
 physical_grasp_verified=False,policy_executed=False,training_executed=False)
write_json(ROOT/'reports/remove_vibration_manifest.json',manifest)
print(json.dumps(published,indent=2))
