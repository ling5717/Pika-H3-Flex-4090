# SPDX-License-Identifier: GPL-3.0-only
"""Verify already-downloaded official workflow components; never downloads files."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", type=Path, required=True)
    a = p.parse_args()
    root = a.model_root.resolve()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "configs/models.json").read_text(encoding="utf8"))
    print(manifest["license_notice"])
    bad = []
    for item in manifest["models"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path escapes model root")
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            print("MISSING/SIZE MISMATCH", item["path"])
            bad.append(item["path"])
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(8 * 1024 * 1024):
                h.update(chunk)
        ok = h.hexdigest() == item["sha256"]
        print("PASS" if ok else "HASH MISMATCH", item["path"])
        if not ok:
            bad.append(item["path"])
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
