"""Read-only publication checks; standard library only, no simulator or optimizer."""
import argparse,ast,hashlib,json,re,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path);args=p.parse_args()
files=(root/'publication/files.txt').read_text().splitlines();manifest=json.loads((root/'publication/content_manifest.json').read_text());checks={};issues=[]
checks['listed_files_present']=all((root/f).is_file() for f in files)
checks['no_symlink_publication']=all(not (root/f).is_symlink() for f in files)
checks['file_size_under_100_MiB']=all((root/f).stat().st_size<100*2**20 for f in files if (root/f).is_file())
bad=[]
for name,expected in manifest['files'].items():
    path=root/name
    h=hashlib.sha256()
    if path.is_file():
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    if not path.is_file() or h.hexdigest()!=expected['sha256']:bad.append(name)
checks['content_hashes_match']=not bad;issues.extend('content hash: '+f for f in bad)
for name in files:
    path=root/name
    if path.suffix=='.py':
        try:ast.parse(path.read_text(),filename=name)
        except (SyntaxError,UnicodeError) as exc:issues.append('syntax: '+str(exc))
    if path.suffix=='.sh':
        result=subprocess.run(['bash','-n',str(path)],capture_output=True,text=True)
        if result.returncode:issues.append('shell: '+name+': '+result.stderr)
checks['python_and_shell_syntax']=not any(s.startswith(('syntax:','shell:')) for s in issues)
broken=[]
for name in files:
    if not name.endswith('.md'):continue
    path=root/name
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
        target=target.split(' "',1)[0].strip('<>').split('#',1)[0]
        if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target):continue
        if not (path.parent/target).resolve().exists():broken.append(name+' -> '+target)
checks['relative_markdown_links_resolve']=not broken;issues.extend('link: '+v for v in broken)
media=json.loads((root/'media/manifest.json').read_text())['videos']
checks['video_manifest_matches']=all((root/v['path']).is_file() and hashlib.sha256((root/v['path']).read_bytes()).hexdigest()==v['sha256'] and (root/v['poster']).is_file() and v['frames']==v['decoded_frames'] for v in media)
for item in json.loads((root/'publication/input_manifest.json').read_text())['files']:
    if hashlib.sha256((root/item['path']).read_bytes()).hexdigest()!=item['sha256']:issues.append('input: '+item['path'])
checks['input_hashes_match']=not any(v.startswith('input:') for v in issues)
secret=[]
for name in files:
    path=root/name
    if path.suffix not in ['.md','.json','.py','.sh','.txt']:continue
    text=path.read_text(errors='replace')
    if re.search(r'gh[pousr]_[A-Za-z0-9]{24,}',text) or re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',text):secret.append(name)
checks['no_obvious_embedded_credentials']=not secret;issues.extend('credential-like content: '+v for v in secret)
if (root/'.git').is_dir():
    tracked=subprocess.check_output(['git','-C',str(root),'ls-files'],text=True).splitlines()
    checks['git_files_within_allowlist']=set(tracked)<=set(files)
    checks['git_root_is_custom_data']=subprocess.check_output(['git','-C',str(root),'rev-parse','--show-toplevel'],text=True).strip()==str(root)
report={'status':'PASS' if all(checks.values()) and not issues else 'FAIL','checks':checks,'issues':issues,'selected_files':len(files),'videos':len(media),'validation_scope':'Static paths/hashes/links/syntax and recorded decode evidence only; no new simulation or training.'}
if args.report:
    out=args.report.resolve()
    if root not in out.parents:raise ValueError('Report must remain inside this repository')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));sys.exit(0 if report['status']=='PASS' else 1)
