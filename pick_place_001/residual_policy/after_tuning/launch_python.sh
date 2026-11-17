#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_repo="${RESMIMIC_ROOT:-$(cd "$task_root/../../../../.." && pwd)}"
task_python="${RESMIMIC_PYTHON:-$task_repo/thirdparty/miniconda3/envs/rl-motion/bin/python}"
if [[ "$task_python" != /* ]]; then
    task_python=$(command -v -- "$task_python")
fi
if [[ ! -x "$task_python" ]]; then
    echo "Missing Isaac Gym Python: $task_python. Set RESMIMIC_PYTHON to its interpreter path." >&2
    exit 1
fi
task_env=$(cd -- "$(dirname -- "$task_python")/.." && pwd)
export PATH="$task_env/bin:$PATH"
if [[ -x "$task_env/bin/x86_64-conda-linux-gnu-c++" ]]; then
    export CXX="$task_env/bin/x86_64-conda-linux-gnu-c++"
fi
export CPLUS_INCLUDE_PATH="$task_env/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$task_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1
exec "$task_python" -u -I -B "$@"
