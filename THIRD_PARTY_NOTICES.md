# Third-party notices

The top-level Apache-2.0 license applies to project code where compatible with the notices retained in individual files. It does not relicense third-party datasets, models, geometry, simulator software or pretrained weights.

- **ResMimic**: [project repository](https://github.com/badriawan/ResMimic), pinned in `configs/dependency_pins.json`; root code license Apache-2.0. Its vendored `legged_gym`, `rsl_rl`, `pose` and other components retain their own notices, including BSD-style licenses.
- **GMR**: [YanjieZe/GMR](https://github.com/YanjieZe/GMR), MIT code license; robot asset notices remain upstream.
- **Mink / MuJoCo**: external dependencies used for IK and reference geometry/rendering. Refer to their upstream licenses.
- **NVIDIA Isaac Gym Preview 4**: external proprietary SDK, obtained and installed separately under NVIDIA's terms. No SDK binary is included.
- **InterPose-derived motion/clock records and floorlamp geometry**: included or referenced for this user-generated research experiment. No additional third-party dataset or geometry redistribution license is granted by the code license. Retain provenance and obtain any permissions required by the originating asset/data provider.
- **Pretrained base policy**: read from pinned ResMimic and copied locally by `tools/prepare_assets.py`; omitted from this Git publication's binary files. Upstream conditions apply.
- **Historical residual checkpoint and simulator/reference videos**: experimental outputs derived from the above dependencies. They document the recorded experiment, including failure cases, and do not provide a new license to underlying assets.

This package is not an official GMR, ResMimic, InterPose or NVIDIA release. Names identify the components used in the experiment.
