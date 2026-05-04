#!/usr/bin/env bash
set -euo pipefail

python "$(dirname "$0")/fieldline_compare.py" "$@"
