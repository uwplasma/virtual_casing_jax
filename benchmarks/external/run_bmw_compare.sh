#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
BMW benchmark scaffold.

Expected workflow:
  1. Build or locate the BMW reference implementation.
  2. Run BMW on a VMEC equilibrium.
  3. Compare vector potential and magnetic-field grids with the JAX BMW prototype.
  4. Report divB, field relative L2, Poincare comparison, runtime, and memory.

This becomes actionable after the JAX BMW prototype lands.
MSG
