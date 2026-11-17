"""Create the explicitly selected GitHub repository and push reviewed local files."""
import argparse,json,os,re,shlex,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',required=True,help='Explicit OWNER/NAME destination')
parser.add_argument('--visibility',required=True,choices=['public','private'])
args=parser.parse_args()
if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',args.repo):parser.error('Use OWNER/NAME')
gh=['bash',str(root/'tools/github.sh')]
def call(command,check=True):return subprocess.run(command,cwd=root,text=True,capture_output=True,check=check)
def git(*a,check=True):return call(['git','-C',str(root),*a],check=check)
if not (root/'.git').is_dir():raise SystemExit('Prepare and review the independent local Git repository first.')
if git('status','--porcelain').stdout.strip():raise SystemExit('Local tracked files are not clean; review/commit changes before publishing.')
auth=call(gh+['auth','status'],check=False)
if auth.returncode:raise SystemExit('GitHub authentication is required. Run: bash tools/github.sh auth login --hostname github.com --git-protocol https --web')
call([sys.executable,str(root/'tools/check_publication.py')])
url='https://github.com/'+args.repo
remote=git('remote','get-url','origin',check=False)
remote_url=remote.stdout.strip()
if remote_url.endswith('.git'):remote_url=remote_url[:-4]
if remote.returncode==0 and remote_url!=url:raise SystemExit('Existing origin points elsewhere; refusing to change it.')
status_path=root/'publication/github_status.json';previous=json.loads(status_path.read_text())
existing=call(gh+['repo','view',args.repo,'--json','nameWithOwner,visibility,url'],check=False)
if existing.returncode==0:
    if previous.get('repository')!=args.repo or previous.get('state')!='repository_created':raise SystemExit('The destination already exists and is not this recorded publication; refusing to overwrite it.')
    repo_info=json.loads(existing.stdout)
    if repo_info['visibility'].lower()!=args.visibility:raise SystemExit('Existing visibility differs; refusing to change access.')
else:
    created=call(gh+['repo','create',args.repo,'--'+args.visibility,'--description','InterPose-to-G1 custom data, reproducible ResMimic experiments, commands and result videos.'],check=False)
    if created.returncode:raise SystemExit(created.stderr or created.stdout)
if remote.returncode:git('remote','add','origin',url+'.git')
source_commit=git('rev-parse','HEAD').stdout.strip()
status={'state':'repository_created','repository':args.repo,'url':url,'visibility':args.visibility,'source_commit':source_commit,'note':'Repository creation recorded here. The publish helper verifies remote branch SHA before reporting push success; see its receipt.'}
status_path.write_text(json.dumps(status,indent=2)+'\n')
git('add','--','publication/github_status.json')
if git('diff','--cached','--quiet',check=False).returncode:
    git('commit','-m','Record GitHub publication destination')
helper='!bash '+shlex.quote(str(root/'tools/github.sh'))+' auth git-credential'
environment=dict(os.environ,GIT_TERMINAL_PROMPT='0')
push=subprocess.run(['git','-C',str(root),'-c','credential.helper=','-c','credential.helper='+helper,'push','--set-upstream','origin','main'],env=environment,text=True,capture_output=True)
if push.returncode:raise SystemExit('Repository exists but push failed:\n'+push.stderr)
local=git('rev-parse','HEAD').stdout.strip()
remote_sha=call(gh+['api','repos/'+args.repo+'/commits/main','--jq','.sha']).stdout.strip()
if local!=remote_sha:raise SystemExit('Push verification failed: remote branch differs.')
receipt={'status':'PUBLISHED_AND_VERIFIED','repository':args.repo,'url':url,'visibility':args.visibility,'commit':local}
(root/'publication/last_push_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
