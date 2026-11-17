#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if command -v gh >/dev/null 2>&1; then
  exec gh "$@"
fi
for task_gh in "$task_root"/.tools/github-cli/*/bin/gh; do
  if [[ -x "$task_gh" ]]; then
    exec "$task_gh" "$@"
  fi
done
echo "GitHub CLI is missing. Install it from https://cli.github.com/ and retry." >&2
exit 1
