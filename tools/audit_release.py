# SPDX-License-Identifier: GPL-3.0-only
"""Check the staged public payload, not local model/cache directories."""
import argparse
import re
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deny-string", action="append", default=[])
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM", "-z"], cwd=root).decode().split("\0")
    errors = []
    secret = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{40,}|AKIA[A-Z0-9]{16}|https?://[^/\s:@]+:[^/\s@]+@)")
    for name in filter(None, paths):
        path = root / name
        if (name.startswith(("benchmarks/local/", "benchmarks/raw/", ".local/"))
                or path.suffix.lower() in (".safetensors", ".ckpt", ".pt", ".pth", ".whl", ".pem", ".key")
                or path.name in (".env", "private-state.json", "ov.conf", "id_rsa")):
            errors.append((name, "forbidden private/binary artifact"))
        blob = subprocess.check_output(["git", "show", ":" + name], cwd=root)
        if len(blob) > 25 * 1024 * 1024:
            errors.append((name, "large file requires explicit release review"))
        try:
            text = blob.decode("utf8")
        except UnicodeDecodeError:
            continue
        if secret.search(text):
            errors.append((name, "credential-like content"))
        for value in a.deny_string:
            if value.casefold() in text.casefold():
                errors.append((name, "machine-private deny string"))
    for name, reason in errors:
        print("FAIL", name, reason)
    if errors:
        raise SystemExit(1)
    print("PASS staged privacy/artifact checks; still review diffs and generated media manually")


if __name__ == "__main__":
    main()
