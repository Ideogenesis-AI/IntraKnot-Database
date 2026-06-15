#!/usr/bin/env python3
"""Verify that SHA-256 digests recorded in registry.yaml match the actual files.

Zero external dependencies — uses the Python standard library only.

Exit codes:
  0 — all entries verified successfully
  1 — one or more entries are missing or have a mismatched digest
"""

import hashlib
import re
import sys
from pathlib import Path

REGISTRY_NAME = "registry.yaml"


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_scripts(text: str) -> list[dict]:
    """Return a list of {file, sha256} dicts extracted from registry.yaml.

    Parses only the subset of YAML used by this registry: mappings whose
    values are single-line scalars, and sequences of such mappings.  The
    format is under our control and well-defined, so a purpose-built parser
    is preferable to an external dependency.
    """
    scripts: list[dict] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect the start of a script list item, e.g. "      - name: foo"
        m = re.match(r"^(\s+)-\s+(\w+):\s*(.*)", line)
        if m:
            block_indent = len(m.group(1))
            block: dict[str, str] = {m.group(2): m.group(3).strip()}
            i += 1
            while i < len(lines):
                cont = lines[i]
                if not cont.strip():
                    i += 1
                    continue
                # Stop when we return to the list-item indentation level or above
                cur_indent = len(cont) - len(cont.lstrip())
                if cur_indent <= block_indent:
                    break
                cm = re.match(r"^\s+(\w+):\s*(.*)", cont)
                if cm:
                    block[cm.group(1)] = cm.group(2).strip()
                i += 1
            if "file" in block and "sha256" in block:
                scripts.append({"file": block["file"], "sha256": block["sha256"]})
        else:
            i += 1
    return scripts


def verify(root: Path) -> bool:
    registry_path = root / REGISTRY_NAME
    if not registry_path.exists():
        print(f"ERROR: {REGISTRY_NAME} not found at {registry_path}")
        return False

    scripts = parse_scripts(registry_path.read_text())
    if not scripts:
        print(f"WARNING: no script entries found in {REGISTRY_NAME}")
        return True

    ok = True
    for entry in scripts:
        file_rel: str = entry["file"]
        expected: str = entry["sha256"]
        file_path = root / file_rel

        if not file_path.exists():
            print(f"  MISSING   {file_rel}")
            ok = False
            continue

        actual = compute_sha256(file_path)
        if actual == expected:
            print(f"  OK        {file_rel}")
        else:
            print(f"  MISMATCH  {file_rel}")
            print(f"            expected : {expected}")
            print(f"            actual   : {actual}")
            ok = False

    return ok


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    print(f"Verifying registry against files in {root}\n")
    passed = verify(root)
    if passed:
        print("\nRegistry OK — all SHA-256 digests match.")
    else:
        print("\nRegistry verification FAILED — see mismatches above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
