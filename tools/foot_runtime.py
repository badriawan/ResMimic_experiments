"""Scoped paths for the clip-specific published foot-correction experiment."""
import os,sys
from pathlib import Path
WORK=Path(__file__).resolve().parents[1]
REPO=Path(os.environ.get('RESMIMIC_ROOT',WORK.parents[1])).expanduser().resolve()
def install():
    sys.dont_write_bytecode=True
    for key,folder in [('TMPDIR','.cache/tmp'),('XDG_CACHE_HOME','.cache'),('MPLCONFIGDIR','.cache/matplotlib')]:
        p=WORK/folder;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    def check(path):
        if not isinstance(path,(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(path)).resolve()
        if p!=Path('/dev/null') and p!=WORK and WORK not in p.parents:raise PermissionError('Write outside custom-data experiment repository: '+str(p))
    def audit(event,args):
        if event=='open':
            mode,flags=args[1:3]
            if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):check(args[0])
        elif event in ('os.mkdir','os.remove','os.rmdir','os.chmod','os.utime'):check(args[0])
        elif event in ('os.rename','os.link','os.symlink'):check(args[0]);check(args[1])
    sys.addaudithook(audit)
