"""Generate the explicit Git allowlist and content manifest from curated paths."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
patterns=['README.md','LICENSE','THIRD_PARTY_NOTICES.md','.gitattributes','docs/*.md','docs/*.html','requirements/*.txt','configs/*.json','tools/*.py','tools/*.sh','tests/*.py','media/*.mp4','media/*.json','media/posters/*.jpg','inputs/*/*','publication/*.json','publication/checks/*.json','converter/scripts/*.py','pick_place_001/scripts/foot_reference_loader.py']
pick='pick_place_001/demo_object_base_policy/'
for stage in ['step_2_contact','step_2_contact/arm_smoothing','step_3_transition']:
    patterns.extend([pick+stage+'/scripts/*.py',pick+stage+'/assets/*'])
task='pick_place_001/residual_policy/after_tuning/'
patterns.extend([task+'assets/*',task+'scripts/*.py',task+'tests/*.py',task+'*.sh',task+'README.md',task+'reports/latest_training_curves.json',task+'reports/latest_training_curves.png'])
fixed=[]
for ref in json.loads((root/'publication/reference_catalog.json').read_text())['references']:
    fixed.extend([ref['motion'],ref['object_motion'],ref['mesh']])
    parent=Path(ref['motion']).parent
    for name in ['contact_plan.json','transition_clock.npz','builder_report.json']:
        if (root/parent/name).is_file():fixed.append(str(parent/name))
for stage in ['step_2_contact','step_2_contact/arm_smoothing']:
    for name in ['reference_validation.json','input_provenance.json','package_manifest.json']:
        p=pick+stage+'/reports/'+name
        if (root/p).exists():fixed.append(p)
fixed.extend([pick+'step_3_transition/reports/validation_A_frame0000_validation.json',pick+'step_3_transition/reports/loader_A_frame0000.json',pick+'step_3_transition/reports/startup_comparison.json'])
historical=task+'runs/20261114T121525_338256Z/'
fixed.extend(historical+name for name in ['model_5000.pt','env_config.json','train_config.json','provenance.json','progress.json','training_validation.json','initialization_checks.json'])
evaluation=task+'evaluations/model_5000_20261114T225204_685687Z/'
fixed.extend(evaluation+name for name in ['result.json','samples.json','artifact_validation.json'])
fixed.extend(['pick_place_001/data_ready_for_policy/p1_attempt09/measurements.json','pick_place_001/data_ready_for_policy/p1_attempt09/diagnostics.npz'])
selected={p.relative_to(root).as_posix() for pattern in patterns for p in root.glob(pattern) if p.is_file()}
for name in fixed:
    if not (root/name).is_file():raise FileNotFoundError(name)
    selected.add(name)
selected.update(['.gitignore','publication/files.txt','publication/content_manifest.json'])
# Machine-updated validation/publishing state is tracked, but excluded from
# content hashes to avoid a circular report -> manifest -> report dependency.
dynamic={'publication/validation.json','publication/github_status.json','publication/content_manifest.json'}
lines=['# Curated publication: every tracked file must be explicitly listed.','# Runtime outputs, SDKs, caches, state, auth tools and backups stay local.','*','!*/','/.verification/','/.tools/','/.runtime/','/.cache/','/archive/','/outputs/']
lines.extend('!/'+name for name in sorted(selected))
(root/'.gitignore').write_text('\n'.join(lines)+'\n')
(root/'publication/files.txt').write_text('\n'.join(sorted(selected))+'\n')
files={}
for name in sorted(selected-dynamic):
    p=root/name
    if not p.is_file():raise FileNotFoundError(name)
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    files[name]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
manifest={'schema_version':1,'files':files,'hash_exclusions':sorted(dynamic),'notes':['This is a selected experiment publication, not a backup of every local file.','Local original datasets/runs and ignored archive backups remain on disk.','Code and artifact provenance/license scopes are documented separately.']}
(root/'publication/content_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'selected_files':len(selected),'content_hashed_files':len(files),'total_MiB':round(sum(v['bytes'] for v in files.values())/2**20,2),'largest_MiB':round(max(v['bytes'] for v in files.values())/2**20,2)}))
