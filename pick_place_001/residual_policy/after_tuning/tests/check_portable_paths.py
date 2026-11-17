"""Standard-library checks for portable metadata and cold-checkout selection."""
import json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT,REPO,relocate_path,relocate_values,sha,write_json,python_executable
import render_policy

checks={}
with patch.dict('os.environ',{'RESMIMIC_PYTHON':sys.executable}):
    checks['interpreter_override']=python_executable()==Path(sys.executable).absolute()
run=ROOT/'runs/20261114T121525_338256Z'
if (run/'provenance.json').is_file():
    provenance=json.loads((run/'provenance.json').read_text())
    checks['historical_input_rebase_hashes']=all(sha(relocate_path(path))==digest for path,digest in provenance['input_sha256'].items())
    config=relocate_values(json.loads((run/'env_config.json').read_text()))
    checks['historical_asset_rebase']=Path(config['asset']['file']).is_file()
    checks['historical_motion_rebase']=Path(config['motion']['motion_file']).is_file() and Path(config['motion']['object_motion_file']).is_file()
with tempfile.TemporaryDirectory(prefix='portable_paths_',dir=str(ROOT/'tests')) as temp:
    root=Path(temp);folder=root/'runs/20261117T000000Z';folder.mkdir(parents=True)
    checkpoint=folder/'model_20.pt';checkpoint.write_bytes(b'fixture, not a trained model')
    (folder/'progress.json').write_text(json.dumps({'status':'COMPLETED','checkpoint':str(checkpoint)}))
    with patch.object(render_policy,'ROOT',root):
        checks['cold_checkout_without_state']=render_policy.resolve_checkpoint()==checkpoint
        checks['explicit_checkpoint_selection']=render_policy.resolve_checkpoint(checkpoint)==checkpoint
checks['upstream_root_rebase']=relocate_path('/old/host/ResMimic/assets/g1/example.urdf')==str(REPO/'assets/g1/example.urdf')
report=dict(status='PASS_PORTABLE_PATH_FIXTURES' if all(checks.values()) else 'FAIL',checks=checks,simulator_executed=False,optimizer_updates=0,training_executed=False,
            limitations=['Path/configuration fixtures only; no GPU rendering, simulator construction, or clean-machine dependency installation is performed.'])
write_json(ROOT/'reports/publication_portable_paths.json',report);print(json.dumps(report,indent=2));assert all(checks.values()),checks
