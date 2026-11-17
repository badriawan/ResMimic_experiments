# Object-aware residual policy after reference tuning

This task uses the selected pick/place Step3 reference, a free physical4kg floorlamp, a frozen ResMimic base and a learned residual actor. See the [complete experiment guide](../../../docs/EXPERIMENTS.md) and [installation](../../../docs/INSTALLATION.md).

The published trainer fixes stale timeout metadata on ordinary transitions after an episode ends. The included historical5000-iteration checkpoint predates that fix and fails physical lifting/carrying; see [results](../../../docs/RESULTS.md). No corrected long-run success is claimed.

From this directory, after configuring RESMIMIC_PYTHON and RESMIMIC_ROOT:

```bash
bash run_training.sh start --num-envs 4 --iterations 20
bash run_training.sh status
tail -f state/training_latest.log
```

After the shortrun completes and checks pass:

```bash
bash run_training.sh start --num-envs 32 --iterations 5000
```

TensorBoard starts automatically on localhost6006. Forward that port on remote servers. `stop` requests a training checkpoint at an iteration boundary; `stop-all` also stops the owned TensorBoard. No extra nohup/ampersand is required. Every start creates a newrun; automaticresume is not implemented.

```bash
bash run_training.sh stop
bash run_training.sh stop-all
bash render_latest.sh
```

The renderer loads the saved residual/critic normalizers and frozenbase. `--base-only` produces a clearly labeled zero-residual physical rollout. New recordings go to evaluations/. Runtime outputs go to runs/, state/, logs/ and localcaches; only selected historicalartifacts are published.

Actor/base/critic feature widths are1764/1757/1741. The base receives stripped raw observations, the residual receives its own normalized observations, and the optimizer excludes base and teacher. Objecttracking rewardweight is2. Future features are fullymasked in this configuration. Initial robot/object reference velocities follow the exact synchronizedhold.

Tests: `tests/check_cpu.py`, `tests/check_timeout_transition.py`, `tests/check_portable_paths.py`, and `scripts/test_background_job.py`. Recorded publication checks are collected in [publication validation](../../../publication/validation.json). They distinguish CPU/mocked validation from any actualphysical or optimizer checks performed.
