# ResMimic Experiments: Custom Data to G1

Reproducible InterPose-to-G1 reference conversion, contact correction, arm smoothing, synchronized startup transitions, and residual-policy experiments for **ResMimic**.

**Current result:** conversion and reference validation work; the recorded base-policy and 5,000-iteration residual experiments **do not successfully lift and carry the floorlamp**. The historical residual run also used a stale-timeout bug. This repository fixes that transition-information bug and includes a regression test. That fix alone does not establish physical manipulation success.

Start with the [installation guide](docs/INSTALLATION.md), then follow the [numbered experiment guide](docs/EXPERIMENTS.md). Every stage identifies its input, command, expected output, and relevant video.

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

## Quick start

After [installation](docs/INSTALLATION.md), from this repository root:

```bash
bash tools/run_geometry.sh doctor
bash tools/run_geometry.sh inspect --dataset pick_place_001
bash tools/run_geometry.sh align --dataset pick_place_001 --output outputs/pick_place_run01
bash tools/run_geometry.sh retarget --dataset pick_place_001 --output outputs/pick_place_run01
bash tools/run_geometry.sh validate --motion outputs/pick_place_run01/motion.pkl
bash tools/run_geometry.sh render \
  --motion outputs/pick_place_run01/motion.pkl \
  --object-mesh outputs/pick_place_run01/floorlamp.obj \
  --output outputs/pick_place_run01/reference.mp4
```

Choose a new output directory for a second attempt; existing scientific outputs are not silently overwritten. For `lift_hold_001`, use the same commands with `--dataset lift_hold_001` and a different output directory. See the [complete workflow](docs/EXPERIMENTS.md) for corrected references, physical base-policy evaluation, background training, TensorBoard, and checkpoint rendering.

## Watch the recorded results

| Stage | Preview / MP4 | Interpretation |
|---|---|---|
| GMR retargeting | [![GMR reference](media/posters/01_gmr_reference.jpg)](media/01_gmr_reference.mp4) | Kinematic reference replay |
| Contact and smoothing | [![Arm smoothing](media/posters/04_arm_smoothing_comparison.jpg)](media/04_arm_smoothing_comparison.mp4) | Kinematic comparison; no grasp-force proof |
| Base policy with free object | [![Base-policy physics](media/posters/06_base_policy_free_object.jpg)](media/06_base_policy_free_object.mp4) | Real PhysX rollout; carrying fails |
| Residual policy, iteration 5,000 | [![Historical residual](media/posters/07_residual_5000_historical.jpg)](media/07_residual_5000_historical.mp4) | Historical buggy-timeout training result; lamp falls |

The [video gallery](docs/VIDEOS.md) lists all nine clips. To use a browser video player locally:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open `http://localhost:8000/docs/videos.html`. This serves existing files and starts no simulator or training process.

## Read the results honestly

The historical residual completed 5,000 PPO updates and preserved the frozen base checkpoint. In its deterministic physical evaluation, it reached 270 control steps / 5.40 seconds, but lamp clearance never exceeded 7.26 mm and final tilt was about 89 degrees. Training completion is not task success. [Detailed results and the timeout defect](docs/RESULTS.md).

The published trainer now emits a fresh timeout snapshot for every transition. Begin with the short smoke experiment before another long run. Corrected-training results must be evaluated independently and must not be confused with the historical video.

## Licensing and attribution

The code license is [Apache-2.0](LICENSE), subject to retained file notices. Data, geometry, pretrained checkpoints, simulator components, and videos have separate provenance; the code license does not relicense those third-party assets. See [third-party notices](THIRD_PARTY_NOTICES.md) and [data provenance](docs/DATA.md).

Please cite [ResMimic](https://github.com/badriawan/ResMimic) and [GMR](https://github.com/YanjieZe/GMR) when using the corresponding methods. This is a custom experimental package built on those projects, not their official benchmark release.
