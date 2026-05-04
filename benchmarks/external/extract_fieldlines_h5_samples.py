"""Extract compact benchmark samples from STELLOPT/FIELDLINES HDF5 output."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from fieldline_compare import load_fieldline_samples


ROOT = Path(__file__).resolve().parents[2]


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return result.stdout.strip()


def extract_samples(args: argparse.Namespace) -> dict[str, object]:
    source = args.source.resolve()
    samples = load_fieldline_samples(source)

    payload = dict(samples)
    payload["metadata_json"] = np.asarray(
        json.dumps(
            {
                "source_format": "STELLOPT/FIELDLINES HDF5",
                "source_h5": str(source),
                "source_input": str(args.input.resolve()) if args.input is not None else None,
                "source_coils": str(args.coils.resolve()) if args.coils is not None else None,
                "source_command": args.source_command,
                "virtual_casing_jax_commit": _git_commit(ROOT),
                "notes": (
                    "Compact section-major Poincare/connection-length sample extracted "
                    "from a real FIELDLINES HDF5 run. The raw HDF5 is intentionally not "
                    "committed because it is much larger than the derived benchmark sample."
                ),
            },
            sort_keys=True,
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)

    report = {
        "status": "completed",
        "source": str(source),
        "out": str(args.out),
        "arrays": {name: list(np.asarray(value).shape) for name, value in samples.items()},
        "sample_count": int(samples.get("poincare_rphiz", np.empty((0, 3))).shape[0]),
        "connection_length_count": int(samples.get("connection_lengths", np.empty(0)).shape[0]),
        "virtual_casing_jax_commit": _git_commit(ROOT),
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="STELLOPT/FIELDLINES .h5/.hdf5 output")
    parser.add_argument("--out", type=Path, required=True, help="Compact .npz sample output")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON extraction report")
    parser.add_argument("--input", type=Path, default=None, help="FIELDLINES input namelist used for the run")
    parser.add_argument("--coils", type=Path, default=None, help="FIELDLINES coils file used for the run")
    parser.add_argument("--source-command", default="", help="Command used to generate the HDF5 file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = extract_samples(parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
