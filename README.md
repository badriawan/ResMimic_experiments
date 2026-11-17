# ResMimic Experiments: Custom Data to G1

Reproducible InterPose-to-G1 reference conversion, contact correction, arm smoothing, synchronized startup transitions, and residual-policy experiments for **ResMimic**.

**Current result:** conversion and reference validation work; the recorded base-policy and 5,000-iteration residual experiments **do not successfully lift and carry the floorlamp**. The historical residual run also used a stale-timeout bug. This repository fixes that transition-information bug and includes a regression test. That fix alone does not establish physical manipulation success.

This README contains the full setup, numbered experiment workflow, commands, data formats, recorded results, all nine video previews, troubleshooting, validation scope, and publication instructions. The individual files in `docs/` remain available as focused guides.

## Contents

- [Repository contents and supported layout](#repository-contents-and-supported-layout)
- [Installation](#installation)
- [Step-by-step experiments](#step-by-step-experiments)
- [All nine experiment videos](#all-nine-experiment-videos)
- [Data formats and provenance](#data-formats-and-provenance)
- [Selected reference files and implementation](#selected-reference-files-and-implementation)
- [Results and known limitations](#results-and-known-limitations)
- [Validation performed](#validation-performed)
- [Troubleshooting](#troubleshooting)
- [Publishing and updating the repository](#publishing-and-updating-the-repository)
- [Licensing and attribution](#licensing-and-attribution)

<a id="repository-contents-and-supported-layout"></a>

## What is included

| Component | Purpose |
|---|---|
| `inputs/` | Minimal original InterPose `.npy` data, clock records, rest-object geometry, and a clip-specific correction seed |
| `tools/` | Portable inspection, alignment, GMR retargeting, validation, reference rendering, and publication checks |
| `pick_place_001/` | Selected foot/contact/smoothed/transition references, research code, physical assets, and the residual experiment |
| `lift_hold_001/` | Selected conversion and smoothed reference artifacts for the second dataset |
| `media/` | Nine selected reference/result videos with English captions and SHA-256 provenance |
| `docs/` | Setup, commands, data schemas, results, limitations, and troubleshooting |
| `publication/` | Explicit publication file list, dependency pins, input hashes, and validation evidence |

Large caches, temporary files, PID state, event logs, redundant attempts, and intermediate checkpoints remain local and are excluded from Git. The repository is an experiment package, not a backup of every local run.

## Supported installation layout

Install this repository at:

```text
ResMimic/
├── GMR/                       # pinned external dependency
├── assets/, legged_gym/, pose/, rsl_rl/
└── InterPose/
    └── custom_data/            # this repository
```

Pinned dependencies:

- ResMimic: `0d54d07d8552484b9a37579c8e08e93edd59a0f5`
- GMR: `25f56db79ac411258b377b6d4ccba0c8ff3372ac`
- Geometry environment: Python 3.11; simulation environment: Python 3.8 with Isaac Gym Preview 4.

Geometry conversion does not require Isaac Gym or a trained policy. Physical evaluation and residual training require an NVIDIA GPU and the simulation environment. GMR, simulator binaries, SMPL body models, and third-party environment installations are not bundled.

<a id="installation"></a>

## Installation

Use Linux x86-64. The commands below keep geometry tools separate from the legacy Isaac Gym environment. Run installation commands yourself in the intended environment; the publication process does not install or accept simulator licenses for you.

### 1. Clone pinned dependencies and this repository

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

### 2. Geometry / GMR environment

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

### 3. Simulation environment

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

### 4. Prepare the frozen base checkpoint

```bash
python3 tools/prepare_assets.py
```

This verifies the base policy already supplied by pinned ResMimic and copies it to the local experiment path. It performs no download, training, or SDK installation. Expected SHA-256:

```text
fd98a2af49b8c1b3e39a57b503272db0dacc0d56c7e68d1c8dddfbd9cb9d931c
```

### 5. Run simulation checks before training

```bash
TASK=pick_place_001/residual_policy/after_tuning
bash "$TASK/launch_python.sh" "$PWD/$TASK/scripts/train_residual.py" --check-only
bash "$TASK/launch_python.sh" "$PWD/$TASK/tests/check_timeout_transition.py"
bash "$TASK/launch_python.sh" "$PWD/$TASK/tests/check_cpu.py"
```

The first command checks inputs/configuration. The latter commands exercise CPU fixtures and actual runner/network interfaces; they do not prove physical grasp success. Then use the short GPU preflight/training commands in [EXPERIMENTS.md](#step-by-step-experiments).

### Environment variables in each terminal

```bash
export RESMIMIC_ROOT=/absolute/path/to/ResMimic
export GMR_PYTHON=/absolute/path/to/geometry/environment/bin/python
export RESMIMIC_PYTHON=/absolute/path/to/simulation/environment/bin/python
cd "$RESMIMIC_ROOT/InterPose/custom_data"
```

The historical fixed interpreter path is only a fallback. New commands should use these explicit overrides when reproducing on another machine. Do not point the simulation launcher at the geometry interpreter.

<a id="step-by-step-experiments"></a>

## Step-by-step experiments

Complete [installation](#installation) first. Commands below run from this repository root, `ResMimic/InterPose/custom_data`, with `GMR_PYTHON`, `RESMIMIC_PYTHON` and `RESMIMIC_ROOT` set. Each scientific attempt uses a new name/output directory.

The main experiment is `pick_place_001`. A second section covers `lift_hold_001`. Do not substitute its differently scaled data into the pick/place-specific correction scripts.

### 1. Inspect the original input

```bash
bash tools/run_geometry.sh doctor
bash tools/run_geometry.sh inspect --dataset pick_place_001
```

Input: `inputs/pick_place_001/motion.npy` and `clock.npz`. Inspect units, quaternion convention, frame counts, original clocks, and joint metadata. The NPY is a trusted dictionary of human and object motion, not G1 joint commands.

Expected: 127 source frames at 30 Hz. Stop if units, joint hierarchy or object geometry differ from the dataset configuration.

### 2. Align and retarget through GMR

```bash
bash tools/run_geometry.sh align --dataset pick_place_001 --output outputs/pick_place_run01
bash tools/run_geometry.sh retarget --dataset pick_place_001 --output outputs/pick_place_run01
bash tools/run_geometry.sh validate \
  --motion outputs/pick_place_run01/motion.pkl \
  --report outputs/pick_place_run01/validation.json
bash tools/run_geometry.sh render \
  --motion outputs/pick_place_run01/motion.pkl \
  --object-mesh outputs/pick_place_run01/floorlamp.obj \
  --output outputs/pick_place_run01/reference.mp4
```

Outputs: `aligned.npz`, `qpos.npy`, `motion.pkl`, paired `motion.npz`, `floorlamp.obj`, IK configuration and reports. Expected common-clock length: **126 frames, 30 Hz, 4.166667 s between endpoints**.

The exporter composes human local rotations through the parent hierarchy, applies one common scene scale, solves GMR/Mink IK, reorders 29 robot joints by name, and calculates the 38-link skeleton with target URDF FK. No physics or policy runs here.

Recorded reference: [Video 01](#video-01). The initial GMR motion still has poor floor/contact geometry; native schema/joint-limit PASS is not approval for training.

### 3. Reproduce the clip-specific foot correction

```bash
"$GMR_PYTHON" -I -B tools/correct_pick_place_feet.py \
  --motion "$PWD/outputs/pick_place_run01/motion.pkl" \
  --aligned "$PWD/outputs/pick_place_run01/aligned.npz" \
  --output-dir "$PWD/outputs/pick_place_feet01"

bash tools/run_geometry.sh validate \
  --motion outputs/pick_place_feet01/interpose_floorlamp_g1_foot_contact.pkl \
  --report outputs/pick_place_feet01/schema_validation.json
```

Expected: 501 frames at 120 Hz, a paired object NPZ, `measurements.json`, solver traces and stance diagnostics. Inspect every gate in `measurements.json`; do not continue with a failed candidate.

This correction is deliberately specific to the bundled 126-frame source. Its inferred swing schedule and earlier upper-body/root seed are supplied in the repository; the seed preserves a prior 7.5 cm pelvis-height correction. The script is not a universal contact estimator for arbitrary new motions. It never claims measured foot forces or dynamic balance.

The selected historical accepted result is also available directly at:

```text
pick_place_001/data_ready_for_policy/p1_attempt09/interpose_floorlamp_g1_foot_contact.pkl
pick_place_001/data_ready_for_policy/p1_attempt09/interpose_floorlamp_g1_foot_contact.npz
```

Recorded result: [Video 02](#video-02).

### 4. Inspect and branch the coupled hand/object correction

The accepted hand/object reference is supplied so you can reproduce downstream experiments without repeating the original parameter search:

```bash
STEP2="$PWD/pick_place_001/demo_object_base_policy/step_2_contact"
CONTACT="$STEP2/reference_corrected/interpose_floorlamp_g1_hand_contact.pkl"

"$GMR_PYTHON" -I -B "$STEP2/scripts/validate_reference.py" \
  --candidate "$CONTACT" --rate 600 --tag user_contact_validation01

"$GMR_PYTHON" -I -B "$STEP2/scripts/render_reference.py" \
  --candidate "$CONTACT" --output "$STEP2/results/user_contact_reference01"
```

Inputs include the paired PKL/NPZ, `contact_plan.json`, source colliders, and G1 URDF. The reference uses opposing fixed-palm patches, a consistent object-frame contact target and no active gripper.

To create an experimental alternative from the foot-contact reference:

```bash
PICK_PLACE_FOOT_REFERENCE="$PWD/outputs/pick_place_feet01/interpose_floorlamp_g1_foot_contact.pkl" \
"$GMR_PYTHON" -I -B "$STEP2/scripts/solve_contact_reference.py" \
  --name user_contact01 --self-seed --patch side
```

Validate the new candidate independently before using it. A solver candidate is not automatically accepted, and changing the grip/path parameters can violate collision or joint limits. The supplied accepted reference is a checkpoint from the earlier coupled/temporal tuning sequence; the short alternative command does not claim to repeat every historical optimization decision.

Recorded accepted reference: [Video 03](#video-03).

### 5. Inspect or regenerate arm smoothing

```bash
ARMS="$STEP2/arm_smoothing"
SMOOTH="$ARMS/reference_smoothed/interpose_floorlamp_g1_hand_contact.pkl"

"$GMR_PYTHON" -I -B "$ARMS/scripts/validate_reference.py" \
  --candidate "$SMOOTH" --rate 2400 --tag user_smoothed_validation01
```

To generate another smoothing candidate from the accepted contact checkpoint:

```bash
export PICK_PLACE_CONTACT_REFERENCE="$CONTACT"
"$GMR_PYTHON" -I -B "$ARMS/scripts/smooth_arms.py" \
  --name user_smoothing01 --iterations 15 \
  --collision-weight 100000 --tangent-weight 150
```

Then validate `"$ARMS/candidates/user_smoothing01/interpose_floorlamp_g1_hand_contact.pkl"` with a new tag. The objective reduces arm acceleration/jerk while constraining contacts and preserving root/legs/waist/object paths. The published accepted result used a previous smoothing candidate as its seed; a new unseeded attempt can produce different values and must pass its own checks.

Recorded comparison: [Video 04](#video-04).

### 6. Generate and validate the shared startup transition

```bash
STEP3="$PWD/pick_place_001/demo_object_base_policy/step_3_transition"
export PICK_PLACE_SMOOTHED_REFERENCE="$SMOOTH"

"$GMR_PYTHON" -I -B "$STEP3/scripts/build_transition.py" \
  --name user_transition01 --start-frame 0 --hold 0.75 --ramp 1.0

"$GMR_PYTHON" -I -B "$STEP3/scripts/validate_transition.py" \
  --candidate "$STEP3/candidates/user_transition01/interpose_floorlamp_g1_transition.pkl" \
  --rate 2400 --tag user_transition_validation01
```

The selected transition is also supplied at `step_3_transition/data_final_reference_transition/`. It has **2,601 frames at 480 Hz**, a 0.75 s exact stationary prefix, a 1 s source-clock ramp and an exact 1× source suffix. `transition_clock.npz` records the mapping. Both robot and lamp follow that mapping; resetting only robot velocity would not create a consistent reference.

Recorded reference replay: [Video 05](#video-05). A smooth reference clock does not eliminate all source pose-derivative discontinuities or prove physical stability.

### 7. Evaluate the frozen base with the free floorlamp

Prepare the base checkpoint first, then render the supplied historical experiment configuration with residual actions disabled:

```bash
python3 tools/prepare_assets.py
TASK="$PWD/pick_place_001/residual_policy/after_tuning"
HISTORICAL="$TASK/runs/20261114T121525_338256Z/model_5000.pt"

bash "$TASK/render_latest.sh" --checkpoint "$HISTORICAL" --base-only
```

The renderer restores the saved environment configuration, rebases historical paths to your checkout, and runs a single bounded deterministic physical rollout. No optimizer update occurs. The supplied residual checkpoint provides compatible run metadata; residual actions are zero in `--base-only` mode.

Outputs appear in a new `"$TASK/evaluations/"` folder, including `base_policy_physx.mp4`, state arrays and `result.json`. This new rendering uses the saved training environment configuration; it is not asserted to be byte-identical to the earlier Step 3 probe, whose termination settings differ.

Recorded Step 3 free-object base-policy result: [Video 06](#video-06). Startup improved, but carrying failed.

### 8. Check the corrected residual pipeline

```bash
bash "$TASK/launch_python.sh" "$TASK/scripts/train_residual.py" --check-only
bash "$TASK/launch_python.sh" "$TASK/tests/check_timeout_transition.py"
bash "$TASK/launch_python.sh" "$TASK/tests/check_cpu.py"
bash "$TASK/launch_python.sh" "$TASK/scripts/train_residual.py" preflight --num-envs 4
```

The timeout regression must pass the endpoint → ordinary transition case. GPU preflight performs forward inference, a physical step and subset reset, but no PPO optimizer update. A PASS is pipeline evidence, not lift/carry evidence.

Residual input: 1,764 features including current object pose. Frozen-base input: the original raw 1,757 features. Critic input: 1,741 features. Saved residual/critic normalizers are separate from the base's embedded normalization. Future features are fully masked in this configuration. The object is a free 4 kg actor, with no weld or support force.

### 9. Start short background training and TensorBoard

```bash
bash "$TASK/run_training.sh" start --num-envs 4 --iterations 20
bash "$TASK/run_training.sh" status
tail -f "$TASK/state/training_latest.log"
```

TensorBoard starts automatically at **http://localhost:6006**. On a remote server, forward port 6006 through your editor or SSH. The training/TensorBoard processes are detached from the terminal; do not add another `&` or `nohup`.

Wait for training to stop, then inspect its `training_validation.json`, `progress.json`, losses and error curves. A 20-iteration test is intended to cross episode boundaries if the policy survives long enough. If it fails, inspect `failure.json` and the log before continuing.

### 10. Start a longer experimental run

After the short run completes successfully:

```bash
bash "$TASK/run_training.sh" start --num-envs 32 --iterations 5000
```

Default rollout batch: 32 × 24 samples, split into three minibatches of 256. Each invocation starts a new residual; it does not resume the historical buggy run. Existing owned TensorBoard can be reused. For another port, add `--port 6007`. For less VRAM, try `--num-envs 16`.

```bash
bash "$TASK/run_training.sh" stop       # training only; cooperative iteration-boundary save
bash "$TASK/run_training.sh" stop-all   # training and its TensorBoard
```

SIGTERM is given up to 30 seconds for a cooperative save before forced termination. A new checkpoint cannot be guaranteed after a crash/forced kill. No automatic resume or reboot service is installed.

Outputs: `runs/<timestamp>/model_<iteration>.pt`, `training.log`, `tensorboard/events.*`, configurations, provenance, update metrics, progress and completion/failure records. Runtime outputs are ignored by the publication allowlist unless you explicitly curate a result.

### 11. Render the latest trained checkpoint

```bash
bash "$TASK/render_latest.sh"
```

Or specify a checkpoint explicitly:

```bash
bash "$TASK/render_latest.sh" --checkpoint "$HISTORICAL"
```

The latter reproduces the historical failed 5,000-iteration policy, even though the current trainer's timeout metadata has been repaired. Loading its saved weights does not turn it into a corrected-training checkpoint. [Video 07](#video-07).

Read `result.json`: lift height/floor clearance, tilt, contact indicators, tracking error, stop reason and input hashes. Do not treat episode survival or a finite reward as grasp/carry success.

### 12. Repeat conversion for the lift/hold dataset

```bash
bash tools/run_geometry.sh inspect --dataset lift_hold_001
bash tools/run_geometry.sh align --dataset lift_hold_001 --output outputs/lift_hold_run01
bash tools/run_geometry.sh retarget --dataset lift_hold_001 --output outputs/lift_hold_run01
bash tools/run_geometry.sh validate --motion outputs/lift_hold_run01/motion.pkl
bash tools/run_geometry.sh render \
  --motion outputs/lift_hold_run01/motion.pkl \
  --object-mesh outputs/lift_hold_run01/floorlamp.obj \
  --output outputs/lift_hold_run01/reference.mp4
```

This dataset uses scale 1.0 and a different recorder clock. Its accepted smoothed lowering pair is supplied under `lift_hold_001/data_tuning_remove_vibration/v2_final/`; the original correction/smoothing research sources are retained in `converter/scripts/` for provenance. Those legacy tools have additional scene/attempt dependencies and are not the portable quick-start entry point.

[Smoothed reference Video 08](#video-08) and [free-object base-policy Video 09](#video-09) show different kinds of evidence. The published residual task is currently configured specifically for the selected pick/place Step 3 pair; do not point it at arbitrary lift/hold data without adapting and validating its configuration and initialization contract.

### Publication validation scope

See [publication validation](publication/validation.json) for the actual checks run on the prepared package, including any clean-checkout tests and bounded physical smoke tests. A fresh dependency installation and successful long corrected training must not be inferred from CPU/mocked checks.

<a id="all-nine-experiment-videos"></a>

## All nine experiment videos

Every published video is shown below with its preview, dataset, recording type, duration, and interpretation. Click a preview to open the repository MP4, or use **Watch / download MP4** for the video file directly. GitHub may open its file viewer rather than play a clip inline.

For a local browser video player, run this command from the repository root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open **http://localhost:8000/docs/videos.html**. This serves the existing videos; it starts no simulator or training process. Reference replay shows stored poses. A PhysX rollout shows policy actions with a free physical object.

<a id="video-01"></a>

### 01. Initial GMR reference

[![Initial GMR reference](media/posters/01_gmr_reference.jpg)](media/01_gmr_reference.mp4)

[Watch / download MP4](https://github.com/badriawan/ResMimic_experiments/raw/refs/heads/main/media/01_gmr_reference.mp4) · **Dataset:** `pick_place_001` · **Type:** Kinematic reference · **Duration:** 4.20 s

Initial GMR retargeting; no policy or physics.

<a id="video-02"></a>

### 02. Foot-contact corrected reference

[![Foot-contact corrected reference](media/posters/02_foot_contact_reference.jpg)](media/02_foot_contact_reference.mp4)

[Watch / download MP4](https://github.com/badriawan/ResMimic_experiments/raw/refs/heads/main/media/02_foot_contact_reference.mp4) · **Dataset:** `pick_place_001` · **Type:** Kinematic reference · **Duration:** 4.17 s

Foot-contact corrected reference; no learned-policy success claim.

<a id="video-03"></a>

### 03. Coupled hand–floorlamp reference

[![Coupled hand–floorlamp reference](media/posters/03_hand_contact_reference.jpg)](media/03_hand_contact_reference.mp4)

[Watch / download MP4](https://github.com/badriawan/ResMimic_experiments/raw/refs/heads/main/media/03_hand_contact_reference.mp4) · **Dataset:** `pick_place_001` · **Type:** Kinematic reference · **Duration:** 4.20 s

Opposing fixed-palm reference contact on the lamp collider.

<a id="video-04"></a>

### 04. Arm smoothing comparison

[![Arm smoothing comparison](media/posters/04_arm_smoothing_comparison.jpg)](media/04_arm_smoothing_comparison.mp4)

[Watch / download MP4](https://github.com/badriawan/ResMimic_experiments/raw/refs/heads/main/media/04_arm_smoothing_comparison.mp4) · **Dataset:** `pick_place_001` · **Type:** Kinematic comparison · **Duration:** 4.18 s

Before/after constrained arm smoothing; remaining vibration can occur.

<a id="video-05"></a>

### 05. Synchronized startup transition

[![Synchronized startup transition](media/posters/05_transition_reference.jpg)](media/05_transition_reference.mp4)

[Watch / download MP4](https://github.com/badriawan/ResMimic_experiments/raw/refs/heads/main/media/05_transition_reference.mp4) · **Dataset:** `pick_place_001` · **Type:** Kinematic reference · **Duration:** 5.43 s

Selected A: 0.75 s hold and 1.0 s shared robot/object clock ramp.

<a id="video-06"></a>

### 06. Base policy with a free floorlamp

[![Base policy with a free floorlamp](media/posters/06_base_policy_free_object.jpg)](media/06_base_policy_free_object.mp4)

[Watch / download MP4](https://github.com/badriawan/ResMimic_experiments/raw/refs/heads/main/media/06_base_policy_free_object.mp4) · **Dataset:** `pick_place_001` · **Type:** PhysX base policy · **Duration:** 5.42 s

Startup improves; physical lift/carry still fails.

<a id="video-07"></a>

### 07. Historical residual policy: 5,000 iterations

[![Historical residual policy: 5,000 iterations](media/posters/07_residual_5000_historical.jpg)](media/07_residual_5000_historical.mp4)

[Watch / download MP4](https://github.com/badriawan/ResMimic_experiments/raw/refs/heads/main/media/07_residual_5000_historical.mp4) · **Dataset:** `pick_place_001` · **Type:** PhysX base plus residual · **Duration:** 5.42 s

Historical 5000-iteration run with the timeout bug; lamp falls. Not a corrected-training result.

<a id="video-08"></a>

### 08. Lift/hold: smoothed reference

[![Lift/hold: smoothed reference](media/posters/08_lift_hold_smoothed_reference.jpg)](media/08_lift_hold_smoothed_reference.mp4)

[Watch / download MP4](https://github.com/badriawan/ResMimic_experiments/raw/refs/heads/main/media/08_lift_hold_smoothed_reference.mp4) · **Dataset:** `lift_hold_001` · **Type:** Kinematic reference · **Duration:** 12.73 s

Smoothed lift/hold/lowering reference; replay is not a physical success test.

<a id="video-09"></a>

### 09. Lift/hold: base policy with a free floorlamp

[![Lift/hold: base policy with a free floorlamp](media/posters/09_lift_hold_free_object.jpg)](media/09_lift_hold_free_object.mp4)

[Watch / download MP4](https://github.com/badriawan/ResMimic_experiments/raw/refs/heads/main/media/09_lift_hold_free_object.mp4) · **Dataset:** `lift_hold_001` · **Type:** PhysX base policy · **Duration:** 12.70 s

Free-object floorlamp experiment; manipulation is not certified successful.

<a id="data-formats-and-provenance"></a>

## Data formats and provenance

### Two different NPZ roles

The source InterPose NPY is a trusted, project-generated dictionary containing human joint positions, local axis-angle rotations, object poses and metadata. Retargeting does not consist of changing a filename extension.

```text
InterPose motion.npy + clock.npz
          |
          v
aligned.npz (human and object at a common clock)
          |                            |
          v                            v
GMR/Mink IK -> qpos.npy       object coordinate transform
          |                            |
          v                            v
motion.pkl (G1 robot)         motion.npz (floorlamp)
```

Never load an arbitrary untrusted pickle/NPY object dictionary. The input manifest records hashes of the bundled trusted experiment files.

### Exported motion pair

| File / field | Shape | Meaning |
|---|---|---|
| PKL `fps` | scalar | Sampling rate shared by robot/object reference |
| PKL `root_pos` | `[T,3]` | Robot root world position, meters |
| PKL `root_rot` | `[T,4]` | Robot root quaternion, XYZW |
| PKL `dof_pos` | `[T,29]` | G1 joint angles in target URDF joint order, radians |
| PKL `local_body_pos` | `[T,38,3]` | URDF forward-kinematic link positions in pelvis frame |
| PKL `link_body_list` | 38 names | Link order for the FK array |
| NPZ `trans` | `[T,3]` | Object world translation, meters |
| NPZ `rot` | `[T,4]` | Object quaternion, XYZW |
| NPZ `source_time_s` | `[T]` | Original capture clock mapped to exported samples |

The robot/object files must stay paired. The object NPZ does not contain robot motor commands or measured physical grip forces.

`qpos.npy` is a separate intermediate: `[T,36] = root XYZ + root WXYZ + 29 joint angles in GMR XML order`. It is not the final PKL layout. The exporter converts quaternion order and reorders joints by name before saving the final motion pair.

### Dataset-specific differences

| Property | `pick_place_001` | `lift_hold_001` |
|---|---|---|
| Original source length | 127 frames | 383 frames |
| Source rate | 30 Hz | 30 Hz |
| Common-clock export | 126 frames | 382 frames |
| Common scene scale | 0.9 | 1.0 |
| Clock source | Stage-10 measured human / derived object clock map | Recorder reference clock / derived object index clock |
| Origin convention | First aligned pelvis XY | First source pelvis XY |

The common-clock export retains supported overlap only, without inventing out-of-range samples. Positions use linear interpolation; local human rotations and object orientation use SLERP. Human local rotations are composed through the SMPL-X parent tree before IK. The first 22 human bodies provide IK targets; the extra two hand markers are not robot DOFs.

Scale and XY origin changes apply consistently to human targets, the object trajectory, and the object rest mesh. The rest mesh is not silently recentered. The original datasets therefore cannot share a single hardcoded scaling assumption.

### Selected reference checkpoints

The package contains the initial raw conversion, the accepted foot-contact intermediate, the accepted coupled hand/object reference, the accepted arm-smoothed reference, and the selected startup-transition reference. The transition adds a 0.75-second stationary reference and a 1-second clock-rate ramp to both robot and lamp. The selected final reference has 2,601 frames at 480 Hz and endpoint duration 5.416667 seconds.

The lift/hold dataset also includes its selected smoothed lowering reference. These are saved experimental checkpoints, not interchangeable representations of one dataset.

### Ownership and upstream material

- InterPose-generated motions and recording clocks are user-supplied experiment artifacts; their provenance is preserved in `publication/input_manifest.json`.
- The floorlamp rest geometry originated in the InterPose processed-object assets. The code license does not grant new rights to the underlying object dataset.
- Robot models remain in pinned GMR/ResMimic dependencies. Their upstream notices still apply.
- The frozen base checkpoint is obtained from the pinned ResMimic checkout, not relicensed by this package.
- The included historical residual checkpoint is an experimental result derived from those dependencies; it is supplied to reproduce the recorded failed evaluation.
- Videos show reference replay or simulator output; each clip's type is recorded in `media/manifest.json`.

No SDK binaries, Conda environments, human body-model weights, or unrelated scene datasets are included. Consult [THIRD_PARTY_NOTICES.md](#licensing-and-attribution) before redistributing upstream data/assets.

<a id="selected-reference-files-and-implementation"></a>

## Selected reference files and implementation

These are the bundled accepted checkpoints, rather than paths created by a new attempt. Each robot PKL must stay paired with the same-basename object NPZ. The selected pick/place transition is the published residual task's input.

| Checkpoint | Robot PKL and paired object NPZ directory | Frames / rate |
|---|---|---|
| Pick/place: raw GMR | [interpose_floorlamp_g1.pkl](pick_place_001/data_after_conversion/interpose_floorlamp_g1.pkl) / [object NPZ](pick_place_001/data_after_conversion/interpose_floorlamp_g1.npz) | 126 / 30 Hz |
| Pick/place: foot correction | [interpose_floorlamp_g1_foot_contact.pkl](pick_place_001/data_ready_for_policy/p1_attempt09/interpose_floorlamp_g1_foot_contact.pkl) / [object NPZ](pick_place_001/data_ready_for_policy/p1_attempt09/interpose_floorlamp_g1_foot_contact.npz) | 501 / 120 Hz |
| Pick/place: hand/object correction | [interpose_floorlamp_g1_hand_contact.pkl](pick_place_001/demo_object_base_policy/step_2_contact/reference_corrected/interpose_floorlamp_g1_hand_contact.pkl) / [object NPZ](pick_place_001/demo_object_base_policy/step_2_contact/reference_corrected/interpose_floorlamp_g1_hand_contact.npz) | 2,001 / 480 Hz |
| Pick/place: arm smoothing | [interpose_floorlamp_g1_hand_contact.pkl](pick_place_001/demo_object_base_policy/step_2_contact/arm_smoothing/reference_smoothed/interpose_floorlamp_g1_hand_contact.pkl) / [object NPZ](pick_place_001/demo_object_base_policy/step_2_contact/arm_smoothing/reference_smoothed/interpose_floorlamp_g1_hand_contact.npz) | 2,001 / 480 Hz |
| Pick/place: selected final transition | [interpose_floorlamp_g1_transition.pkl](pick_place_001/demo_object_base_policy/step_3_transition/data_final_reference_transition/interpose_floorlamp_g1_transition.pkl) / [object NPZ](pick_place_001/demo_object_base_policy/step_3_transition/data_final_reference_transition/interpose_floorlamp_g1_transition.npz) | 2,601 / 480 Hz |
| Lift/hold: converted | [lift_hold_001_g1.pkl](lift_hold_001/data_after_conversion/final/lift_hold_001_g1.pkl) / [object NPZ](lift_hold_001/data_after_conversion/final/lift_hold_001_g1.npz) | 1,525 / 120 Hz |
| Lift/hold: selected smoothed lowering | [lift_hold_001_g1.pkl](lift_hold_001/data_tuning_remove_vibration/v2_final/lift_hold_001_g1.pkl) / [object NPZ](lift_hold_001/data_tuning_remove_vibration/v2_final/lift_hold_001_g1.npz) | 1,525 / 120 Hz |

### Which code runs each stage?

| Stage | Main implementation |
|---|---|
| Dataset paths, scale and clocks | [configs/datasets.json](configs/datasets.json) |
| Interpreter selection and geometry CLI | [tools/run_geometry.sh](tools/run_geometry.sh), [tools/experiment.py](tools/experiment.py) |
| Foot-contact correction | [tools/correct_pick_place_feet.py](tools/correct_pick_place_feet.py), [tools/foot_contact_support.py](tools/foot_contact_support.py) |
| Coupled hand/object correction | [solve_contact_reference.py](pick_place_001/demo_object_base_policy/step_2_contact/scripts/solve_contact_reference.py) and [validation](pick_place_001/demo_object_base_policy/step_2_contact/scripts/validate_reference.py) |
| Arm smoothing | [smooth_arms.py](pick_place_001/demo_object_base_policy/step_2_contact/arm_smoothing/scripts/smooth_arms.py) and [validation](pick_place_001/demo_object_base_policy/step_2_contact/arm_smoothing/scripts/validate_reference.py) |
| Shared robot/object startup | [build_transition.py](pick_place_001/demo_object_base_policy/step_3_transition/scripts/build_transition.py) and [validation](pick_place_001/demo_object_base_policy/step_3_transition/scripts/validate_transition.py) |
| Base checkpoint verification | [tools/prepare_assets.py](tools/prepare_assets.py) |
| Physical floorlamp environment | [physical_floorlamp_env.py](pick_place_001/residual_policy/after_tuning/scripts/physical_floorlamp_env.py) |
| Residual adapter and timeout repair | [residual_env.py](pick_place_001/residual_policy/after_tuning/scripts/residual_env.py) |
| Residual training entry point | [train_residual.py](pick_place_001/residual_policy/after_tuning/scripts/train_residual.py) |
| Detached training and TensorBoard | [run_training.sh](pick_place_001/residual_policy/after_tuning/run_training.sh), [background_job.py](pick_place_001/residual_policy/after_tuning/scripts/background_job.py) |
| Checkpoint evaluation and video | [render_latest.sh](pick_place_001/residual_policy/after_tuning/render_latest.sh), [render_policy.py](pick_place_001/residual_policy/after_tuning/scripts/render_policy.py) |
| Timeout regression | [check_timeout_transition.py](pick_place_001/residual_policy/after_tuning/tests/check_timeout_transition.py) |
| Legacy lift/hold conversion research | [converter/scripts/](converter/scripts/) — provenance; additional historical dependencies apply |

The residual actor consumes 1,764 features, the frozen base consumes 1,757 raw features, and the critic consumes 1,741 features. The optimizer excludes the base and teacher. Object-tracking reward weight is 2, and future features are fully masked in this configuration. Initial robot and object reference velocities follow the same stationary startup prefix. The renderer restores the saved residual and critic normalizers alongside the frozen base.

The [reference catalog](publication/reference_catalog.json) records the selected PKL/NPZ pairs, meshes, sampling rates, endpoint durations, quaternion convention, and SHA-256 hashes. Use [the content manifest](publication/content_manifest.json) and [input manifest](publication/input_manifest.json) to audit exact published files.

<a id="results-and-known-limitations"></a>

## Results and known limitations

### Supported claims

| Experiment | Result |
|---|---|
| NPY alignment and GMR retargeting | Kinematic reference conversion works; robot/object schemas are readable |
| Foot-contact correction | Selected clip-specific reference passes its recorded kinematic checks |
| Hand/object correction | Contact geometry improves; no physical force-closure guarantee |
| Arm smoothing | Acceleration/jerk are reduced while selected reference constraints are retained |
| Synchronized startup transition | Initial simulation tracking improves; carrying still fails |
| Historical residual training | 5,000 optimizer iterations completed; physical lifting/carrying failed |
| Published timeout repair | Fresh per-transition timeout snapshot and endpoint-to-ordinary regression pass |

Reference replay is generated from stored poses. Physical evaluation uses gravity, collisions, a free rigid-body floorlamp and policy actions. Their videos must not be interpreted as equivalent evidence.

### Historical residual result

Checkpoint: `pick_place_001/residual_policy/after_tuning/runs/20261114T121525_338256Z/model_5000.pt`.

![Historical residual training curves](pick_place_001/residual_policy/after_tuning/reports/latest_training_curves.png)

**Historical run, before the timeout repair.** These curves document the included failed 5,000-iteration experiment; they are not a corrected-training result. [Curve data](pick_place_001/residual_policy/after_tuning/reports/latest_training_curves.json), [training validation](pick_place_001/residual_policy/after_tuning/runs/20261114T121525_338256Z/training_validation.json), and [physical evaluation metrics](pick_place_001/residual_policy/after_tuning/evaluations/model_5000_20261114T225204_685687Z/result.json) are included for inspection.

- Completed optimizer iterations: 5,000.
- Final training reward: approximately 44.77.
- Value loss increased from approximately 3.94 to 10,792,426.
- Deterministic evaluation: 270 control steps / 5.40 seconds / 271 video frames.
- Maximum object floor clearance: 7.26 mm.
- Final lamp tilt: 89.02 degrees.
- Final lamp position error: 1.645 m.
- Final robot root XY error: 0.704 m.
- Supported-airborne proxy samples: zero.
- Object pose setters after evaluation began: zero.

The final 16.7 ms of the 5.416667-second reference is outside the bounded control rollout. Surviving the complete bounded episode is not evidence of successful object manipulation.

### Confirmed timeout defect in that run

The parent environment creates a new `time_out_buf` each step, but the old training adapter only assigned `extras['time_outs']` when an episode reset. After a timeout, the information dictionary could retain `True` on an ordinary nonterminal transition. PPO then added an incorrect bootstrap term to the stored reward.

The actual-method CPU reproduction showed:

```text
ordinary step: done=False, current timeout=False, reported timeout=True
raw reward=1, critic value=10, gamma=0.99
incorrect stored reward=10.9; expected stored reward=1.0
```

The published adapter refreshes the timeout information on every transition, masks it by the actual done flags and clones the snapshot. Its regression covers endpoint → reset → ordinary step, rather than only testing isolated resets. This fixes a confirmed target-corruption defect. The large critic loss is consistent with that defect, but a corrected comparison run is needed to measure its contribution to every physical failure.

The initial preparation checks missed this sequential case. Their earlier PASS status described pipeline completion and file/model integrity, not correct learning targets or task success.

### Remaining physical problems

The hands lose contact before sustained lifting: in the recorded residual evaluation, one hand contact disappears around 2.5 seconds and both recorded net hand-contact norms are zero around 3 seconds. These values are net rigid-body contact indicators, not individual grasp normal-force measurements.

The reference uses fixed hands without an active gripper. Valid surface geometry alone does not prove sufficient contact force, friction margin, actuator torque or balance. Previous force assessments were conditional calculations on an earlier reference; they are not a new feasibility certificate for the current policy.

Object tracking uses a dense but rapidly decaying `exp(-10*d)` reward. At 0.5 m position error, the raw term is only about 0.0067. Once the object slips away, that term supplies a weak recovery signal while robot-tracking rewards remain available.

### Recommended next experiments

1. Run the published timeout regression and a short PPO smoke run; inspect targets, losses and reset behavior.
2. Establish stable physical contact while standing, then lifting in place.
3. Validate a force-feasible hand/object reference and contact/slip behavior before adding walking.
4. Evaluate corrected checkpoints separately from the historical run, with multiple seeds and explicit lift/carry metrics.

No corrected long-run or successful-carry claim should be inferred from the historical checkpoint/video. See `publication/validation.json` for the checks actually performed while preparing this repository.

<a id="validation-performed"></a>

## Validation performed

The recorded [publication validation](publication/validation.json) and [clean-checkout record](publication/clean_checkout.json) distinguish completed checks from work still needed. These checks reused the pinned external dependencies and existing installed environments.

| Completed check | Evidence / scope |
|---|---|
| Both datasets aligned and retargeted with actual GMR | Native schema and joint-limit validation; clip-specific foot correction also exercised |
| Clean-checkout geometry workflow | Conversion, reference rendering, and corrected reference handling |
| Dense hand/object validation | 600 Hz reference sampling |
| Dense arm and transition validation | 2,400 Hz reference sampling; transition builder exercised |
| CPU network/layout fixtures | 21 checks |
| Sequential timeout regression | 7 checks, including endpoint → reset → ordinary transition |
| Portable path fixtures | 8 checks |
| Detached-process mock tests | 6 checks; no actual new TensorBoard server was started for these tests |
| Geometry unit tests | 3 checks |
| Historical residual and base-only PhysX evaluation | Finite recorded rollouts; zero optimizer updates |
| Filesystem tracing of clean-checkout simulation | Zero reads from the original unpublished `custom_data` tree |
| Video validation | All nine MP4s decoded: 2,889 frames in total |
| Publication checks | Allowlist, file sizes, content/input hashes, Python/Bash syntax, relative links, videos, and obvious credential patterns |

**Not established by these checks:** a fresh dependency installation, a new corrected PPO training run, successful long corrected training, or successful physical lift/carry. The historical failure videos remain failure evidence.

For a read-only package check from the repository root:

```bash
python3 tools/check_publication.py
```

Detailed JSON evidence is retained in [publication/checks/](publication/checks/), [advanced geometry checks](publication/clean_advanced_geometry.json), [video validation](publication/video_validation.json), and [the reference environment record](publication/reference_environment.json).

<a id="troubleshooting"></a>

## Troubleshooting

### Geometry commands cannot find GMR or robot assets

Run `bash tools/run_geometry.sh doctor`. Install the repository under the supported `ResMimic/InterPose/custom_data` layout, or set `RESMIMIC_ROOT` to the pinned external checkout. Set `GMR_PYTHON` to the geometry environment's absolute Python path. A complete fresh checkout requires the bundled input files, not only the README/scripts.

### Simulation uses the wrong Python

Set `RESMIMIC_PYTHON=/absolute/path/to/python` for the Python 3.8 Isaac Gym environment. The launcher and its TensorBoard child share that interpreter. Geometry Python 3.11 is a separate environment.

### NumPy pickle compatibility

Some reference PKLs were written by NumPy 2.x while the simulator uses NumPy 1.23. Use the supplied loader adapter and actual ResMimic loader. A plain `pickle.load` in an unrelated Python 3.8 script may fail with `numpy._core`. Do not rewrite the motion or reorder joints just to suppress the import error.

### Output already exists

Choose another attempt name or output directory. Original scientific outputs are preserved. Do not delete the selected reference to rerun an experiment. Examples: `outputs/pick_place_run02`, `--name user_transition02`, or a new validation `--tag`.

### CUDA out of memory

Stop the owned training process before starting another. Reduce `--num-envs` to 16 or 4; retain a minibatch count that divides `num_envs * 24`. The automatic default bounds minibatch samples. Check other GPU processes with `nvidia-smi`; this repository does not kill unrelated GPU jobs.

### TensorBoard port occupied

Use the existing owned TensorBoard port reported by `run_training.sh status`, or select another free port such as `--port 6007`. To change a live owned server's port, use `stop-all` first. Foreign listeners are never terminated. On a remote server, forward the selected port to your local browser.

TensorBoard 2.14's API requires the compatible protobuf range in `requirements/simulation.txt`; a protobuf 5 installation can break dashboard requests even if event files can be read.

### Training says PASS but the lamp falls

Read [RESULTS.md](#results-and-known-limitations). Completion checks prove updates/checkpoints/file integrity, not successful physical manipulation. The included 5,000-iteration run was trained before the timeout repair. A new trainer does not change the weights already saved in that historical checkpoint.

### No simulator window

The supplied recording commands are headless and save MP4 files. Open the video in the reported evaluation folder. This is intentional finite evaluation, not an interactive viewer command.

### Renderer refers to another machine's home directory

Use the published renderer, which rebases paths from saved run metadata to your current ResMimic checkout. Keep the paired references, training configs and checkpoint together at their curated paths. Do not manually replace normalization statistics or silently substitute a different reference.

### Long training was interrupted

`stop` requests an iteration-boundary checkpoint; inspect `interrupted.json` and the reported checkpoint. A forced kill may leave only the last periodic checkpoint. The current launcher starts new runs and does not provide automatic resume. Preserve the run metadata if adding a compatible resume workflow later.

<a id="publishing-and-updating-the-repository"></a>

## Publishing and updating the repository

This repository is prepared independently inside `ResMimic/InterPose/custom_data`; it does not change the parent repository's Git history. The publication uses an explicit allowlist so caches, SDKs, local authentication tools, PID state and repeated checkpoints cannot be added accidentally.

### Review the package

```bash
python3 tools/update_publication.py
python3 tools/check_publication.py
git status --short
```

Inspect `publication/files.txt` and `publication/content_manifest.json`. Update the selection intentionally when adding a new result. Review data/asset provenance and [third-party notices](#licensing-and-attribution); the code license does not relicense external datasets.

### Authenticate

For creating a new repository through the GitHub CLI, the account owner must complete CLI authentication. Do not paste a password or token into a chat, README or committed file:

```bash
bash tools/github.sh auth login --hostname github.com --git-protocol https --web
bash tools/github.sh auth status
```

The wrapper uses an installed GitHub CLI or the local ignored CLI downloaded from its official release during preparation. On another machine, install [GitHub CLI](https://cli.github.com/) first. Authentication is independent of experiment execution.

For an existing repository, Git authentication supplied by an editor, credential manager or SSH agent may already support pushing even when GitHub CLI is not logged in. After reviewing and committing updates, use the existing configured remote:

```bash
git remote -v
git push origin main
git ls-remote origin refs/heads/main
```

Preserve any existing remote history; do not force-push over someone else's work. The initial `ResMimic_experiments` README commit was retained through a merge before uploading this package.

### Create and push

Once the local allowlisted files are committed, explicitly choose the destination and visibility:

```bash
python3 tools/publish_github.py --repo badriawan/ResMimic_experiments --visibility public
```

Use `--visibility private` for a private repository. The helper requires authentication and a clean reviewed working tree, refuses an unrelated existing destination, creates the GitHub repository, pushes `main` and checks the remote commit SHA. A local Git commit alone is not reported as a successful GitHub publication.

The commands use the official [GitHub CLI repository creation interface](https://cli.github.com/manual/gh_repo_create). Videos and the selected historical checkpoint are ordinary Git files below the [GitHub per-file limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github); do not commit all intermediate training checkpoints. Larger future artifacts should be distributed through a deliberate release/LFS workflow.

### Status records

- `publication/github_status.json`: local preparation or repository-creation metadata.
- `publication/last_push_receipt.json`: local mutable receipt generated only after a verified push; excluded from the publication allowlist.
- `publication/validation.json`: preparation checks and their scope, including unexecuted tests.

If credentials or GitHub access are unavailable, the accurate outcome is **local repository prepared; GitHub publication pending authentication**. Do not claim the repository is online until the remote branch check succeeds.

<a id="licensing-and-attribution"></a>

## Licensing and attribution

The top-level Apache-2.0 license applies to project code where compatible with the notices retained in individual files. It does not relicense third-party datasets, models, geometry, simulator software or pretrained weights.

- **ResMimic**: [project repository](https://github.com/badriawan/ResMimic), pinned in `configs/dependency_pins.json`; root code license Apache-2.0. Its vendored `legged_gym`, `rsl_rl`, `pose` and other components retain their own notices, including BSD-style licenses.
- **GMR**: [YanjieZe/GMR](https://github.com/YanjieZe/GMR), MIT code license; robot asset notices remain upstream.
- **Mink / MuJoCo**: external dependencies used for IK and reference geometry/rendering. Refer to their upstream licenses.
- **NVIDIA Isaac Gym Preview 4**: external proprietary SDK, obtained and installed separately under NVIDIA's terms. No SDK binary is included.
- **InterPose-derived motion/clock records and floorlamp geometry**: included or referenced for this user-generated research experiment. No additional third-party dataset or geometry redistribution license is granted by the code license. Retain provenance and obtain any permissions required by the originating asset/data provider.
- **Pretrained base policy**: read from pinned ResMimic and copied locally by `tools/prepare_assets.py`; omitted from this Git publication's binary files. Upstream conditions apply.
- **Historical residual checkpoint and simulator/reference videos**: experimental outputs derived from the above dependencies. They document the recorded experiment, including failure cases, and do not provide a new license to underlying assets.

This package is not an official GMR, ResMimic, InterPose or NVIDIA release. Names identify the components used in the experiment.


The project code license is [Apache-2.0](LICENSE), subject to retained file notices. Please cite [ResMimic](https://github.com/badriawan/ResMimic) and [GMR](https://github.com/YanjieZe/GMR) when using their corresponding methods. Consult the upstream projects for citation metadata and asset-specific terms.
