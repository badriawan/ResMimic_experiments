# Data formats and provenance

## Two different NPZ roles

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

## Exported motion pair

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

## Dataset-specific differences

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

## Selected reference checkpoints

The package contains the initial raw conversion, the accepted foot-contact intermediate, the accepted coupled hand/object reference, the accepted arm-smoothed reference, and the selected startup-transition reference. The transition adds a 0.75-second stationary reference and a 1-second clock-rate ramp to both robot and lamp. The selected final reference has 2,601 frames at 480 Hz and endpoint duration 5.416667 seconds.

The lift/hold dataset also includes its selected smoothed lowering reference. These are saved experimental checkpoints, not interchangeable representations of one dataset.

## Ownership and upstream material

- InterPose-generated motions and recording clocks are user-supplied experiment artifacts; their provenance is preserved in `publication/input_manifest.json`.
- The floorlamp rest geometry originated in the InterPose processed-object assets. The code license does not grant new rights to the underlying object dataset.
- Robot models remain in pinned GMR/ResMimic dependencies. Their upstream notices still apply.
- The frozen base checkpoint is obtained from the pinned ResMimic checkout, not relicensed by this package.
- The included historical residual checkpoint is an experimental result derived from those dependencies; it is supplied to reproduce the recorded failed evaluation.
- Videos show reference replay or simulator output; each clip's type is recorded in `media/manifest.json`.

No SDK binaries, Conda environments, human body-model weights, or unrelated scene datasets are included. Consult [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) before redistributing upstream data/assets.
