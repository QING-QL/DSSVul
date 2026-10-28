import hashlib
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "MANIFEST.sha256"
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Invalid manifest path")
        expected[name] = digest
    errors = []
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append("Missing or changed: " + name)
    extras = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
              and ".git" not in p.relative_to(root).parts
              and p.relative_to(root).as_posix() not in expected
              and p != manifest]
    errors.extend("Not part of the source package: " + name for name in extras)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Source package verified: {len(expected)} files plus the checksum manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
