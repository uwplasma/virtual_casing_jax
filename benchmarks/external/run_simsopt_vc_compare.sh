#!/usr/bin/env bash
set -euo pipefail

python "$(dirname "$0")/simsopt_vc_compare.py" "$@"
