# Step-by-step experiments

Complete [installation](INSTALLATION.md) first. Commands below run from this repository root, `ResMimic/InterPose/custom_data`, with `GMR_PYTHON`, `RESMIMIC_PYTHON` and `RESMIMIC_ROOT` set. Each scientific attempt uses a new name/output directory.

The main experiment is `pick_place_001`. A second section covers `lift_hold_001`. Do not substitute its differently scaled data into the pick/place-specific correction scripts.

## 1. Inspect the original input

```bash
bash tools/run_geometry.sh doctor
bash tools/run_geometry.sh inspect --dataset pick_place_001
```

Input: `inputs/pick_place_001/motion.npy` and `clock.npz`. Inspect units, quaternion convention, frame counts, original clocks, and joint metadata. The NPY is a trusted dictionary of human and object motion, not G1 joint commands.

Expected: 127 source frames at 30 Hz. Stop if units, joint hierarchy or object geometry differ from the dataset configuration.

## 2. Align and retarget through GMR

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

Recorded reference: [Video 01](../media/01_gmr_reference.mp4). The initial GMR motion still has poor floor/contact geometry; native schema/joint-limit PASS is not approval for training.

## 3. Reproduce the clip-specific foot correction

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

Recorded result: [Video 02](../media/02_foot_contact_reference.mp4).

## 4. Inspect and branch the coupled hand/object correction

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

Recorded accepted reference: [Video 03](../media/03_hand_contact_reference.mp4).

## 5. Inspect or regenerate arm smoothing

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

Recorded comparison: [Video 04](../media/04_arm_smoothing_comparison.mp4).

## 6. Generate and validate the shared startup transition

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

Recorded reference replay: [Video 05](../media/05_transition_reference.mp4). A smooth reference clock does not eliminate all source pose-derivative discontinuities or prove physical stability.

## 7. Evaluate the frozen base with the free floorlamp

Prepare the base checkpoint first, then render the supplied historical experiment configuration with residual actions disabled:

```bash
python3 tools/prepare_assets.py
TASK="$PWD/pick_place_001/residual_policy/after_tuning"
HISTORICAL="$TASK/runs/20261114T121525_338256Z/model_5000.pt"

bash "$TASK/render_latest.sh" --checkpoint "$HISTORICAL" --base-only
```

The renderer restores the saved environment configuration, rebases historical paths to your checkout, and runs a single bounded deterministic physical rollout. No optimizer update occurs. The supplied residual checkpoint provides compatible run metadata; residual actions are zero in `--base-only` mode.

Outputs appear in a new `"$TASK/evaluations/"` folder, including `base_policy_physx.mp4`, state arrays and `result.json`. This new rendering uses the saved training environment configuration; it is not asserted to be byte-identical to the earlier Step 3 probe, whose termination settings differ.

Recorded Step 3 free-object base-policy result: [Video 06](../media/06_base_policy_free_object.mp4). Startup improved, but carrying failed.

## 8. Check the corrected residual pipeline

```bash
bash "$TASK/launch_python.sh" "$TASK/scripts/train_residual.py" --check-only
bash "$TASK/launch_python.sh" "$TASK/tests/check_timeout_transition.py"
bash "$TASK/launch_python.sh" "$TASK/tests/check_cpu.py"
bash "$TASK/launch_python.sh" "$TASK/scripts/train_residual.py" preflight --num-envs 4
```

The timeout regression must pass the endpoint → ordinary transition case. GPU preflight performs forward inference, a physical step and subset reset, but no PPO optimizer update. A PASS is pipeline evidence, not lift/carry evidence.

Residual input: 1,764 features including current object pose. Frozen-base input: the original raw 1,757 features. Critic input: 1,741 features. Saved residual/critic normalizers are separate from the base's embedded normalization. Future features are fully masked in this configuration. The object is a free 4 kg actor, with no weld or support force.

## 9. Start short background training and TensorBoard

```bash
bash "$TASK/run_training.sh" start --num-envs 4 --iterations 20
bash "$TASK/run_training.sh" status
tail -f "$TASK/state/training_latest.log"
```

TensorBoard starts automatically at **http://localhost:6006**. On a remote server, forward port 6006 through your editor or SSH. The training/TensorBoard processes are detached from the terminal; do not add another `&` or `nohup`.

Wait for training to stop, then inspect its `training_validation.json`, `progress.json`, losses and error curves. A 20-iteration test is intended to cross episode boundaries if the policy survives long enough. If it fails, inspect `failure.json` and the log before continuing.

## 10. Start a longer experimental run

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

## 11. Render the latest trained checkpoint

```bash
bash "$TASK/render_latest.sh"
```

Or specify a checkpoint explicitly:

```bash
bash "$TASK/render_latest.sh" --checkpoint "$HISTORICAL"
```

The latter reproduces the historical failed 5,000-iteration policy, even though the current trainer's timeout metadata has been repaired. Loading its saved weights does not turn it into a corrected-training checkpoint. [Video 07](../media/07_residual_5000_historical.mp4).

Read `result.json`: lift height/floor clearance, tilt, contact indicators, tracking error, stop reason and input hashes. Do not treat episode survival or a finite reward as grasp/carry success.

## 12. Repeat conversion for the lift/hold dataset

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

[Smoothed reference Video 08](../media/08_lift_hold_smoothed_reference.mp4) and [free-object base-policy Video 09](../media/09_lift_hold_free_object.mp4) show different kinds of evidence. The published residual task is currently configured specifically for the selected pick/place Step 3 pair; do not point it at arbitrary lift/hold data without adapting and validating its configuration and initialization contract.

## Publication validation scope

See [publication validation](../publication/validation.json) for the actual checks run on the prepared package, including any clean-checkout tests and bounded physical smoke tests. A fresh dependency installation and successful long corrected training must not be inferred from CPU/mocked checks.
