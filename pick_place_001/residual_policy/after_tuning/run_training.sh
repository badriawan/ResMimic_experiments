#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 -I -B "$task_root/scripts/background_job.py" "$@"
