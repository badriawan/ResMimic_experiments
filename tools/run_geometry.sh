#!/usr/bin/env bash
set -euo pipefail
task_tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_project=$(cd "$task_tools/.." && pwd)
task_repo=${RESMIMIC_ROOT:-$(cd "$task_project/../.." && pwd)}
if [[ -n ${GMR_PYTHON:-} ]]; then
  task_python=$GMR_PYTHON
elif [[ -x "$task_repo/thirdparty/gmr_omomo/bin/python" ]]; then
  task_python="$task_repo/thirdparty/gmr_omomo/bin/python"
else
  task_python=python3
fi
if ! command -v "$task_python" >/dev/null 2>&1; then
  echo "Geometry Python not found. Set GMR_PYTHON to your Python 3.11 geometry environment." >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$task_python" -u -I -B "$task_tools/experiment.py" "$@"
