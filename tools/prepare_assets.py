"""Copy the pinned upstream base checkpoint locally; no network or training."""
import hashlib,json,os,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1]
repo=Path(os.environ.get('RESMIMIC_ROOT',root.parents[1])).expanduser().resolve()
expected='fd98a2af49b8c1b3e39a57b503272db0dacc0d56c7e68d1c8dddfbd9cb9d931c'
source=repo/'legged_gym/assets/checkpoints/base_policy.pt'
target=root/'pick_place_001/base_policy/base_policy.pt'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if not source.is_file() or digest(source)!=expected:raise SystemExit('Pinned upstream base policy is missing or differs; check ResMimic commit and checkpoint.')
target.parent.mkdir(parents=True,exist_ok=True)
if target.exists():
    if digest(target)!=expected:raise SystemExit('Existing local checkpoint differs; refusing to overwrite.')
else:shutil.copyfile(source,target)
print(json.dumps({'status':'PASS','checkpoint':str(target),'sha256':expected,'network_used':False,'training_executed':False},indent=2))
