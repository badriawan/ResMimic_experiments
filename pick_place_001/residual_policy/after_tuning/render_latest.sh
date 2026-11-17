#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$task_root/launch_python.sh" "$task_root/scripts/render_policy.py" "$@"
