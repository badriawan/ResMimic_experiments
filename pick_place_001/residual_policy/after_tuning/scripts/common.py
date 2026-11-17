"""Artifact boundary for residual training after reference tuning."""
import os,sys,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=Path(os.environ.get('RESMIMIC_ROOT',str(ROOT.parents[4]))).expanduser().resolve()
PICK=ROOT.parents[1]
STEP3=PICK/'demo_object_base_policy/step_3_transition'
MOTION=STEP3/'data_final_reference_transition/interpose_floorlamp_g1_transition.pkl'
OBJECT=MOTION.with_suffix('.npz')
LOCAL_CHECKPOINT=PICK/'base_policy/base_policy.pt'
CHECKPOINT=LOCAL_CHECKPOINT if LOCAL_CHECKPOINT.is_file() else REPO/'legged_gym/assets/checkpoints/base_policy.pt'
ASSETS=ROOT/'assets'
URDF=REPO/'assets/g1/g1_custom_collision_29dof.urdf'

def python_executable():
    """Resolve the configured Isaac Gym interpreter without starting it."""
    value=os.environ.get('RESMIMIC_PYTHON',str(REPO/'thirdparty/miniconda3/envs/rl-motion/bin/python'))
    candidate=Path(value).expanduser()
    if not candidate.is_absolute():
        found=shutil.which(value)
        if found:candidate=Path(found)
    return candidate.absolute()

def relocate_path(value):
    """Rebase historical metadata from its original ResMimic installation."""
    text=str(value)
    current=Path(text)
    custom_root=ROOT.parents[2]
    # Rebasing must be idempotent, including a checkout nested inside another
    # custom_data directory during a clean-tree test.
    if current.is_absolute() and (current==custom_root or custom_root in current.parents):
        return str(CHECKPOINT) if current==LOCAL_CHECKPOINT else text
    marker='/InterPose/custom_data/'
    if marker in text:
        suffix=text.split(marker,1)[1]
        candidate=ROOT.parents[2]/suffix
        # The upstream checkpoint is an equivalent fallback when the local
        # shared base copy is not distributed with the experiment repository.
        if suffix.endswith('pick_place_001/base_policy/base_policy.pt'):
            return str(CHECKPOINT)
        return str(candidate)
    marker='/ResMimic/'
    if marker in text:return str(REPO/text.split(marker,1)[1])
    return text

def relocate_values(value):
    if isinstance(value,str):return relocate_path(value)
    if isinstance(value,list):return [relocate_values(v) for v in value]
    if isinstance(value,dict):return {k:relocate_values(v) for k,v in value.items()}
    return value

def install():
    sys.dont_write_bytecode=True;os.chdir(ROOT)
    for key,folder in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('MPLCONFIGDIR','cache/matplotlib'),('TORCH_EXTENSIONS_DIR','cache/torch_extensions'),('__GL_SHADER_DISK_CACHE_PATH','cache/nvidia'),('MESA_SHADER_CACHE_DIR','cache/mesa')]:
        p=ROOT/folder;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONFAULTHANDLER='1',WANDB_MODE='disabled',WANDB_DISABLED='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    def check(path):
        if not isinstance(path,(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(path)).resolve()
        if p==Path('/dev/null'):return
        if p!=ROOT and ROOT not in p.parents:raise PermissionError('Residual training write outside designated directory: '+str(p))
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
