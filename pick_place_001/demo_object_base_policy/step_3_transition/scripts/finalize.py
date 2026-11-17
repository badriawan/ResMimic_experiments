"""Verify final step3 artifacts and write an integrity manifest, excluding caches."""
import ast,json,sys,subprocess
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,BASELINE,CHECKPOINT,install,sha,write_json

def main():
    install();checks={};sources={}
    for name in ['A_frame0000','B_frame0048']:
        for kind,filename in [('geometry','validation_'+name+'_validation.json'),('loader','loader_'+name+'.json'),('velocity','transition_metrics_'+name+'.json')]:
            report=json.loads((ROOT/'reports'/filename).read_text())
            checks[kind+'_'+name]=report['status'].startswith('PASS') and all(report.get('checks',{}).values())
    comparison=json.loads((ROOT/'reports/startup_comparison.json').read_text())
    checks['all_six_rollout_audits']=all(all(c.values()) for c in comparison['checks'].values())
    for filename in ['interpose_floorlamp_g1_transition.pkl','interpose_floorlamp_g1_transition.npz','contact_plan.json','transition_clock.npz','builder_report.json']:
        checks['selected_exact_'+filename]=sha(ROOT/'reference_transition'/filename)==sha(ROOT/'candidates/A_frame0000'/filename)
    for name in ['videos/comparison_validation.json','videos/reference_transition/video_validation.json']:
        checks[name]=json.loads((ROOT/name).read_text())['status']=='PASS'
    # Recheck every recorded file hash, including previous references and inputs.
    for area in ['reports','results','candidates','videos']:
        for p in (ROOT/area).rglob('*.json'):
            data=json.loads(p.read_text())
            def visit(v):
                if isinstance(v,dict):
                    for k,item in v.items():
                        if k in ['input_sha256','source_sha256','outputs_sha256'] and isinstance(item,dict):
                            for path,digest in item.items():
                                if isinstance(digest,str) and len(digest)==64 and path.startswith('/'):
                                    if path in sources:assert sources[path]==digest,(path,'inconsistent recorded hash')
                                    sources[path]=digest
                        visit(item)
                elif isinstance(v,list):
                    for item in v:visit(item)
            visit(data)
    checks['all_recorded_file_hashes_preserved']=all(Path(p).is_file() and sha(p)==h for p,h in sources.items())
    checks['latest_source_hash']=sha(BASELINE)=='f3905b738068cfa116ae5a8f1ae674c07f9a6e61cbd90dcc5c7fe44634d3af9b'
    checks['latest_object_hash']=sha(BASELINE.with_suffix('.npz'))=='ddb9689312cca554e2a474b5a87b0bf2f8d98bd19d1557e66ed9b727e30c1029'
    checks['base_checkpoint_hash']=sha(CHECKPOINT)=='fd98a2af49b8c1b3e39a57b503272db0dacc0d56c7e68d1c8dddfbd9cb9d931c'
    python_files=sorted((ROOT/'scripts').glob('*.py'))
    for path in python_files:ast.parse(path.read_text(),filename=str(path))
    checks['python_syntax']=True
    shell_files=sorted(ROOT.glob('*.sh'))
    for path in shell_files:subprocess.run(['bash','-n',str(path)],check=True)
    checks['shell_syntax']=True
    checks['architecture_review_recorded']=(ROOT/'reports/architecture_review.md').is_file()
    checks['all_full_runs_reach_bounded_source_endpoint']=all(comparison['runs'][tag]['stop_reason']=='source_clip_complete' for tag in ['full_baseline','full_A','full_B'])
    checks['carry_failure_disclosed']=comparison['selection_scope'].endswith('physical carry remains unsuccessful') and 'Lampu rebah' in (ROOT/'README.md').read_text()
    assert all(checks.values()),checks
    write_json(ROOT/'reports/completion_checks.json',dict(status='PASS_STEP3_INITIAL_TRANSITION',checks=checks,checked_file_hashes=sources,
        python_files=len(python_files),shell_files=len(shell_files),selected_candidate='A_frame0000',physical_carry_success=False,training_executed=False,
        scope='Transition consistency and startup improvement verified; whole physical carry failed and is disclosed.'))
    excluded={'cache','tmp','__pycache__'}
    excluded_files={ROOT/'manifest.json',ROOT/'logs/finalize.log'}
    files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(ROOT.rglob('*')) if p.is_file() and not excluded.intersection(p.relative_to(ROOT).parts) and p not in excluded_files}
    write_json(ROOT/'manifest.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),root=str(ROOT),status='STEP3_COMPLETE_FOR_INITIAL_TRANSITION_ONLY',
        selected_reference='reference_transition/interpose_floorlamp_g1_transition.pkl',physical_carry_success=False,training_executed=False,
        exclusions=['manifest.json itself','logs/finalize.log (active verification stdout)','cache/','tmp/','__pycache__/'],files=files))
    print(json.dumps(dict(status='PASS',checks=len(checks),recorded_hashes=len(sources),manifest_files=len(files))),flush=True)

if __name__=='__main__':main()
