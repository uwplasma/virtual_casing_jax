#!/usr/bin/env python3
"""Extract W7X_ Fourier coefficients from BIEST surface.txx."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def main():
    src = Path("/Users/rogerio/local/virtual-casing/extern/BIEST/include/biest/surface.txx")
    if not src.exists():
        raise SystemExit(f"Missing source file: {src}")

    text = src.read_text()
    pattern = re.compile(
        r"Rbc\(([-\d]+),([-\d]+)\)\s*=\s*\(Real\)\s*([-\d.E+]+);\s*Zbs\(\1,\2\)\s*=\s*\(Real\)\s*([-\d.E+]+);"
    )

    rbc = np.zeros((21, 21), dtype=np.float64)
    zbs = np.zeros((21, 21), dtype=np.float64)

    matches = pattern.findall(text)
    if not matches:
        raise SystemExit("No W7X_ coefficients found")

    for i_s, j_s, r_s, z_s in matches:
        i = int(i_s)
        j = int(j_s)
        r = float(r_s)
        z = float(z_s)
        rbc[i + 10, j + 10] = r
        zbs[i + 10, j + 10] = z

    out = Path("/Users/rogerio/local/virtual_casing_jax/virtual_casing_jax/w7x_coeffs.py")
    with out.open("w") as f:
        f.write('"""W7X_ Fourier coefficients extracted from BIEST surface.txx."""\n')
        f.write("from __future__ import annotations\n\n")
        f.write("import numpy as np\n\n")
        f.write("RBC = np.array(\n")
        f.write(np.array2string(rbc, separator=", ", max_line_width=120))
        f.write(", dtype=np.float64)\n\n")
        f.write("ZBS = np.array(\n")
        f.write(np.array2string(zbs, separator=", ", max_line_width=120))
        f.write(", dtype=np.float64)\n\n")
        f.write("W7X_NFP = 5\n")

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
