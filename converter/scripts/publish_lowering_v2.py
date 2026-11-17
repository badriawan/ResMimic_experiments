"""Publish a validated v2 copy without changing the accepted hold-only pair."""
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from converter_common import ROOT,VIBRATION_DATA,install,sha,write_json
install()
reports={name:ROOT/'reports'/filename for name,filename in dict(geometry='lowering_v2_attempt04_actual600hz_validation.json',loader='lowering_loader_attempt04.json',video='lowering_video_attempt04.json',tail='lowering_v2_tail_audit.json',comparison='lowering_comparison_video_final.json').items()}
r={k:json.loads(p.read_text()) for k,p in reports.items()}
assert r['geometry']['status']=='PASS_LOWERING_REFERENCE' and not r['geometry']['failed_gates']
assert r['loader']['status']=='PASS_ACTUAL_LOADER' and r['loader']['quality_status']=='PASS_SAMPLED_FOOT_CONTACT'
assert r['video']['status']=='PASS_REFERENCE_VIDEO' and all(r['video']['checks'].values())
assert r['comparison']['status']=='PASS_DECODE'
candidate=Path(r['geometry']['motion']);source=Path(r['geometry']['source']);config=Path(r['geometry']['configuration_path'])
assert sha(candidate)==r['geometry']['motion_sha256']==r['loader']['motion_sha256']==r['tail']['motion_sha256']
assert sha(source)==r['geometry']['source_sha256']
assert sha(config)==r['geometry']['configuration_sha256']
assert r['video']['source_files'][str(candidate)]==sha(candidate)
assert all(sha(p)==h for p,h in r['video']['source_files'].items())
assert all(sha(p)==h for p,h in r['geometry']['validation_source_sha256'].items())
assert sha(reports['geometry'])==r['tail']['geometry_validation_sha256']
assert sha(candidate.with_suffix('.npz'))==sha(source.with_suffix('.npz'))
for key in ['video','comparison']:
 p=Path(r[key]['video'] if key=='video' else r[key]['path']);h=r[key]['video_sha256'] if key=='video' else r[key]['sha256'];assert sha(p)==h
out=VIBRATION_DATA/'v2'
if out.exists():raise FileExistsError('Refusing to overwrite a published version')
out.mkdir();published={}
for suffix in ['.pkl','.npz']:
 dst=out/('lift_hold_001_g1'+suffix);shutil.copyfile(candidate.with_suffix(suffix),dst)
 assert sha(dst)==sha(candidate.with_suffix(suffix));published[str(dst)]=sha(dst)
(out/'assets').mkdir();mesh=out/'assets/floorlamp.obj';shutil.copyfile(VIBRATION_DATA/'assets/floorlamp.obj',mesh);published[str(mesh)]=sha(mesh)
files=['smooth_lift_hold_lowering.py','prepare_lowering_quality_profile.py','validate_lowering_vibration.py','validate_lift_hold_loader.py','render_lift_hold_reference.py','review_lowering_vibration.py','audit_lowering_tail.py','publish_lowering_v2.py']
manifest=dict(status='PASS_REFERENCE_KINEMATICS',published=published,validated_candidate=str(candidate),validated_candidate_sha256=sha(candidate),
 source_v1=str(source),source_v1_sha256=sha(source),source_object_sha256=sha(source.with_suffix('.npz')),
 quality_profile=str(config),quality_profile_sha256=sha(config),reports_sha256={str(p):sha(p) for p in reports.values()},
 implementation_sha256={str(ROOT/'scripts'/name):sha(ROOT/'scripts'/name) for name in files},
 generation_report=str(ROOT/'reports/lowering_generation_lowering_v2_attempt04.json'),generation_report_sha256=sha(ROOT/'reports/lowering_generation_lowering_v2_attempt04.json'),
 video=r['video']['video'],video_sha256=r['video']['video_sha256'],comparison_video=r['comparison']['path'],comparison_video_sha256=r['comparison']['sha256'],
 edited_interval_s=r['geometry']['configuration']['edit_start_end_s'],
 lowering_interval_s=r['geometry']['lowering_start_end_s'],limitation='Kinematic reference. Preserve intended stepping; entire accepted hold and object data exact. Exterior resumes at11.5s without a global C1/C2 guarantee. Not physical grasp or policy validation.',
 physics_executed=False,policy_executed=False,training_executed=False)
assert sha(source)==manifest['source_v1_sha256']
write_json(ROOT/'reports/lowering_v2_manifest.json',manifest)
print(json.dumps(dict(status=manifest['status'],published=published),indent=2))
