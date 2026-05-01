#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
STELLOPT EXTENDER benchmark scaffold.

Expected workflow:
  1. Locate or build STELLOPT with EXTENDER.
  2. Run EXTENDER on a matched VMEC/coils input.
  3. Export total, coil-only, and plasma-only field samples.
  4. Compare against B_coils + B_internal^VC from virtual_casing_jax.
  5. Write JSON metrics and plots for point fields and LCFS normal residuals.
MSG
