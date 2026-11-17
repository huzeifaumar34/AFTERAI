"""Validate a dataset without installing the optional training stack."""

from __future__ import annotations

import argparse
import sys

from afterai.data import validate_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to a JSONL chat dataset")
    args = parser.parse_args()

    errors = validate_jsonl(args.dataset)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
