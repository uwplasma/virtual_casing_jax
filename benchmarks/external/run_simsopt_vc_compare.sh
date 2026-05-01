#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
SIMSOPT virtual-casing benchmark scaffold.

Expected workflow:
  1. Create or activate an environment with simsopt, vmec_jax, and virtual_casing_jax.
  2. Load a matched VMEC input/wout case.
  3. Compute simsopt.mhd.VirtualCasing.from_vmec(...).
  4. Compute virtual_casing_jax VirtualCasingExteriorField external branch on the same grid.
  5. Write JSON metrics:
       - relative L2 B_external_normal
       - max relative error
       - source grid and target grid
       - runtime and memory

This scaffold intentionally does not auto-install external codes yet.
MSG
