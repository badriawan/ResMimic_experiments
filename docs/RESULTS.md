# Results and known limitations

## Supported claims

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

## Historical residual result

Checkpoint: `pick_place_001/residual_policy/after_tuning/runs/20261114T121525_338256Z/model_5000.pt`.

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

## Confirmed timeout defect in that run

The parent environment creates a new `time_out_buf` each step, but the old training adapter only assigned `extras['time_outs']` when an episode reset. After a timeout, the information dictionary could retain `True` on an ordinary nonterminal transition. PPO then added an incorrect bootstrap term to the stored reward.

The actual-method CPU reproduction showed:

```text
ordinary step: done=False, current timeout=False, reported timeout=True
raw reward=1, critic value=10, gamma=0.99
incorrect stored reward=10.9; expected stored reward=1.0
```

The published adapter refreshes the timeout information on every transition, masks it by the actual done flags and clones the snapshot. Its regression covers endpoint → reset → ordinary step, rather than only testing isolated resets. This fixes a confirmed target-corruption defect. The large critic loss is consistent with that defect, but a corrected comparison run is needed to measure its contribution to every physical failure.

The initial preparation checks missed this sequential case. Their earlier PASS status described pipeline completion and file/model integrity, not correct learning targets or task success.

## Remaining physical problems

The hands lose contact before sustained lifting: in the recorded residual evaluation, one hand contact disappears around 2.5 seconds and both recorded net hand-contact norms are zero around 3 seconds. These values are net rigid-body contact indicators, not individual grasp normal-force measurements.

The reference uses fixed hands without an active gripper. Valid surface geometry alone does not prove sufficient contact force, friction margin, actuator torque or balance. Previous force assessments were conditional calculations on an earlier reference; they are not a new feasibility certificate for the current policy.

Object tracking uses a dense but rapidly decaying `exp(-10*d)` reward. At 0.5 m position error, the raw term is only about 0.0067. Once the object slips away, that term supplies a weak recovery signal while robot-tracking rewards remain available.

## Recommended next experiments

1. Run the published timeout regression and a short PPO smoke run; inspect targets, losses and reset behavior.
2. Establish stable physical contact while standing, then lifting in place.
3. Validate a force-feasible hand/object reference and contact/slip behavior before adding walking.
4. Evaluate corrected checkpoints separately from the historical run, with multiple seeds and explicit lift/carry metrics.

No corrected long-run or successful-carry claim should be inferred from the historical checkpoint/video. See `publication/validation.json` for the checks actually performed while preparing this repository.
