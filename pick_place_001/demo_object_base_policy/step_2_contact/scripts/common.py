"""Artifact boundary for the coupled hand-object reference correction."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[4]
PICK=ROOT.parents[1]
SOURCE=PICK/'data_ready_for_policy/p1_attempt09'
MOTION=Path(os.environ.get('PICK_PLACE_FOOT_REFERENCE',str(SOURCE/'interpose_floorlamp_g1_foot_contact.pkl'))).expanduser().resolve();OBJECT=MOTION.with_suffix('.npz');MESH=PICK/'assets/floorlamp.obj'
CHECKPOINT=PICK/'base_policy/base_policy.pt'
PREVIOUS=ROOT.parent/'step_0_contact_foot'
PREVIOUS_RUN=PREVIOUS/'results/attempt01'
ASSETS=ROOT/'assets'
URDF=REPO/'assets/g1/g1_custom_collision_29dof.urdf'

def install():
    sys.dont_write_bytecode=True;os.chdir(ROOT)
    for key,folder in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('MPLCONFIGDIR','cache/matplotlib'),('TORCH_EXTENSIONS_DIR','cache/torch_extensions'),('__GL_SHADER_DISK_CACHE_PATH','cache/nvidia'),('MESA_SHADER_CACHE_DIR','cache/mesa')]:
        p=ROOT/folder;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONFAULTHANDLER='1',WANDB_MODE='disabled',WANDB_DISABLED='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    def check(path):
        if not isinstance(path,(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(path)).resolve()
        if p==Path('/dev/null'):return
        if p!=ROOT and ROOT not in p.parents:raise PermissionError('Step2 write outside designated directory: '+str(p))
    def audit(event,args):
        if event=='open':
            mode,flags=args[1:3]
            if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):check(args[0])
        elif event in ('os.mkdir','os.remove','os.rmdir','os.chmod','os.utime'):check(args[0])
        elif event in ('os.rename','os.link','os.symlink'):check(args[0]);check(args[1])
    sys.addaudithook(audit)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
