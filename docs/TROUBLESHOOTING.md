# Troubleshooting

## Geometry commands cannot find GMR or robot assets

Run `bash tools/run_geometry.sh doctor`. Install the repository under the supported `ResMimic/InterPose/custom_data` layout, or set `RESMIMIC_ROOT` to the pinned external checkout. Set `GMR_PYTHON` to the geometry environment's absolute Python path. A complete fresh checkout requires the bundled input files, not only the README/scripts.

## Simulation uses the wrong Python

Set `RESMIMIC_PYTHON=/absolute/path/to/python` for the Python 3.8 Isaac Gym environment. The launcher and its TensorBoard child share that interpreter. Geometry Python 3.11 is a separate environment.

## NumPy pickle compatibility

Some reference PKLs were written by NumPy 2.x while the simulator uses NumPy 1.23. Use the supplied loader adapter and actual ResMimic loader. A plain `pickle.load` in an unrelated Python 3.8 script may fail with `numpy._core`. Do not rewrite the motion or reorder joints just to suppress the import error.

## Output already exists

Choose another attempt name or output directory. Original scientific outputs are preserved. Do not delete the selected reference to rerun an experiment. Examples: `outputs/pick_place_run02`, `--name user_transition02`, or a new validation `--tag`.

## CUDA out of memory

Stop the owned training process before starting another. Reduce `--num-envs` to 16 or 4; retain a minibatch count that divides `num_envs * 24`. The automatic default bounds minibatch samples. Check other GPU processes with `nvidia-smi`; this repository does not kill unrelated GPU jobs.

## TensorBoard port occupied

Use the existing owned TensorBoard port reported by `run_training.sh status`, or select another free port such as `--port 6007`. To change a live owned server's port, use `stop-all` first. Foreign listeners are never terminated. On a remote server, forward the selected port to your local browser.

TensorBoard 2.14's API requires the compatible protobuf range in `requirements/simulation.txt`; a protobuf 5 installation can break dashboard requests even if event files can be read.

## Training says PASS but the lamp falls

Read [RESULTS.md](RESULTS.md). Completion checks prove updates/checkpoints/file integrity, not successful physical manipulation. The included 5,000-iteration run was trained before the timeout repair. A new trainer does not change the weights already saved in that historical checkpoint.

## No simulator window

The supplied recording commands are headless and save MP4 files. Open the video in the reported evaluation folder. This is intentional finite evaluation, not an interactive viewer command.

## Renderer refers to another machine's home directory

Use the published renderer, which rebases paths from saved run metadata to your current ResMimic checkout. Keep the paired references, training configs and checkpoint together at their curated paths. Do not manually replace normalization statistics or silently substitute a different reference.

## Long training was interrupted

`stop` requests an iteration-boundary checkpoint; inspect `interrupted.json` and the reported checkpoint. A forced kill may leave only the last periodic checkpoint. The current launcher starts new runs and does not provide automatic resume. Preserve the run metadata if adding a compatible resume workflow later.
