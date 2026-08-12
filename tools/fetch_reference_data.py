"""Download the optional upstream-reference test data."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import urllib.request


URL = (
    "https://github.com/uwplasma/virtual_casing_jax/releases/download/v0.0.3/"
    "virtual-casing-jax-reference-data-v1.tar.xz"
)
SHA256 = "1cc83063b4b05f73fa7aa40abeff7ee123b26d71a1277c2feeda5cc0f78eae1d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(archive_path: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive_path, "r:xz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if not any(part.startswith("._") for part in Path(member.name).parts)
        ]
        for member in members:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unsafe archive member: {member.name}")
        archive.extractall(root, members=members)  # noqa: S202 - checksum verified above


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--cache", type=Path, default=Path(".cache/reference-data-v1.tar.xz"))
    parser.add_argument("--force", action="store_true", help="download even when the cache is valid")
    args = parser.parse_args()

    if args.force or not args.cache.exists() or _sha256(args.cache) != SHA256:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.cache.with_suffix(args.cache.suffix + ".part")
        with urllib.request.urlopen(URL) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(args.cache)

    actual = _sha256(args.cache)
    if actual != SHA256:
        raise RuntimeError(f"reference-data checksum mismatch: {actual}")

    _extract(args.cache, args.destination)
    print(f"Reference data extracted under {args.destination / 'tests'}")


if __name__ == "__main__":
    main()
