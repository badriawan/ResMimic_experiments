# Installation

Use Linux x86-64. The commands below keep geometry tools separate from the legacy Isaac Gym environment. Run installation commands yourself in the intended environment; the publication process does not install or accept simulator licenses for you.

## 1. Clone pinned dependencies and this repository

```bash
git clone https://github.com/badriawan/ResMimic.git
cd ResMimic
git checkout 0d54d07d8552484b9a37579c8e08e93edd59a0f5
git clone https://github.com/YanjieZe/GMR.git GMR
git -C GMR checkout 25f56db79ac411258b377b6d4ccba0c8ff3372ac
mkdir -p InterPose
git clone https://github.com/badriawan/ResMimic_experiments.git InterPose/custom_data
export RESMIMIC_ROOT="$PWD"
cd InterPose/custom_data
```

If `GMR` or `InterPose/custom_data` already exists, inspect it and reuse the intended checkout; do not clone over existing work. `publication/github_status.json` records the publication state and verified destination of a local preparation.

## 2. Geometry / GMR environment

With Conda already installed:

```bash
conda create -n resmimic-geometry python=3.11 -y
conda activate resmimic-geometry
python -m pip install -r requirements/geometry-tested.txt
python -m pip install -e "$RESMIMIC_ROOT/GMR"
export GMR_PYTHON="$(command -v python)"
```

GMR installation may resolve additional dependencies described by its pinned `pyproject.toml`. The bundled source skeleton already contains positions and local rotations, so this conversion does not reconstruct a human surface or require SMPL body-model weights. See the [pinned GMR project](https://github.com/YanjieZe/GMR/tree/25f56db79ac411258b377b6d4ccba0c8ff3372ac) for its own requirements and licenses.

`geometry-tested.txt` records the core versions used for the publication checks; it is not an exhaustive transitive environment lock. `geometry.txt` contains broader compatibility ranges for an explicitly revalidated alternative environment.

Check the geometry environment:

```bash
bash tools/run_geometry.sh doctor
bash tools/run_geometry.sh inspect --dataset pick_place_001
python tools/check_publication.py
```

`GMR_PYTHON` is an absolute interpreter path. If it is unset, the local development layout also recognizes `ResMimic/thirdparty/gmr_omomo/bin/python`. Do not use Python 3.8 for GMR's geometry environment.

## 3. Simulation environment

Isaac Gym Preview 4 is an external NVIDIA dependency. Obtain/install it under its applicable terms; this repository does not redistribute the SDK. Follow [NVIDIA's Isaac Gym page](https://developer.nvidia.com/isaac-gym) and the pinned ResMimic instructions. A typical environment setup, after making the SDK available locally, is:

```bash
conda create -n resmimic-sim python=3.8 -y
conda activate resmimic-sim
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
export ISAAC_GYM_PATH=/absolute/path/to/isaacgym
python -m pip install -e "$ISAAC_GYM_PATH/python"
python -m pip install -e "$RESMIMIC_ROOT/rsl_rl" -e "$RESMIMIC_ROOT/pose" -e "$RESMIMIC_ROOT/legged_gym"
python -m pip install -r requirements/simulation.txt
export RESMIMIC_PYTHON="$(command -v python)"
```

Use a driver compatible with the chosen CUDA/PyTorch build. The recorded physical experiments ran on an NVIDIA RTX 4070 Ti SUPER with 16 GB VRAM. Default training uses 32 environments and small minibatches; actual VRAM availability still matters.

System prerequisites include a working C++ compiler, EGL/OpenGL libraries and DejaVu fonts for recorded captions. On a minimal Ubuntu installation these may require installing `build-essential`, `libegl1`, `libgl1` and `fonts-dejavu-core` through your system package manager. Do not install the SDK or accept third-party terms automatically through an unreviewed helper.

TensorBoard 2.14 uses `protobuf<5` in the pinned ResMimic requirements. The historical server had protobuf 5.29.6 installed despite that constraint; the new requirements follow the dependency constraint instead of reproducing that mismatch.

## 4. Prepare the frozen base checkpoint

```bash
python3 tools/prepare_assets.py
```

This verifies the base policy already supplied by pinned ResMimic and copies it to the local experiment path. It performs no download, training, or SDK installation. Expected SHA-256:

```text
fd98a2af49b8c1b3e39a57b503272db0dacc0d56c7e68d1c8dddfbd9cb9d931c
```

## 5. Run simulation checks before training

```bash
TASK=pick_place_001/residual_policy/after_tuning
bash "$TASK/launch_python.sh" "$PWD/$TASK/scripts/train_residual.py" --check-only
bash "$TASK/launch_python.sh" "$PWD/$TASK/tests/check_timeout_transition.py"
bash "$TASK/launch_python.sh" "$PWD/$TASK/tests/check_cpu.py"
```

The first command checks inputs/configuration. The latter commands exercise CPU fixtures and actual runner/network interfaces; they do not prove physical grasp success. Then use the short GPU preflight/training commands in [EXPERIMENTS.md](EXPERIMENTS.md).

## Environment variables in each terminal

```bash
export RESMIMIC_ROOT=/absolute/path/to/ResMimic
export GMR_PYTHON=/absolute/path/to/geometry/environment/bin/python
export RESMIMIC_PYTHON=/absolute/path/to/simulation/environment/bin/python
cd "$RESMIMIC_ROOT/InterPose/custom_data"
```

The historical fixed interpreter path is only a fallback. New commands should use these explicit overrides when reproducing on another machine. Do not point the simulation launcher at the geometry interpreter.
