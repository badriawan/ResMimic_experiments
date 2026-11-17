"""Paths and write boundary for the lift_hold_001 conversion stages."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
SOURCE = ROOT.parent / 'lift_hold_001'
DATA = SOURCE / 'data_after_conversion'
VIBRATION_DATA = SOURCE / 'data_remove_vibration'


def motion_root(path):
    path = Path(path).resolve()
    for root in (DATA, VIBRATION_DATA):
        if root in path.parents:
            return root
    raise ValueError('Motion path outside conversion output directories')


def install():
    sys.dont_write_bytecode = True
    os.chdir(ROOT)
    for key, folder in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),
                        ('MPLCONFIGDIR','cache/matplotlib'),('TORCH_EXTENSIONS_DIR','cache/torch_extensions'),
                        ('__GL_SHADER_DISK_CACHE_PATH','cache/nvidia')]:
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',WANDB_DISABLED='true',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')

    def check(path):
        if not isinstance(path, (str,bytes,os.PathLike)):
            return
        p = Path(os.fsdecode(path)).resolve()
        if p == Path('/dev/null'):
            return  # discard sink, never a generated artifact
        if not any(p == root or root in p.parents for root in (ROOT, DATA, VIBRATION_DATA)):
            raise PermissionError('Conversion write outside converter/data_after_conversion: ' + str(p))

    def audit(event,args):
        if event == 'open':
            mode, flags = args[1],args[2]
            if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
                check(args[0])
        elif event in ('os.mkdir','os.remove','os.rmdir','os.chmod','os.utime'):
            check(args[0])
        elif event in ('os.rename','os.link','os.symlink'):
            check(args[0]); check(args[1])
    sys.addaudithook(audit)
    return ROOT


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
