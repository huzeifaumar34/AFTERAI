"""Command-line entry point for AFTERAI."""

from __future__ import annotations

import argparse
import sys

from afterai import __version__
from afterai.data import validate_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afterai",
        description="Tools for the AFTERAI research project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("info", help="show project and runtime information")

    validate_parser = subparsers.add_parser(
        "validate-data",
        help="validate a JSONL supervised fine-tuning dataset",
    )
    validate_parser.add_argument("path", help="path to the JSONL dataset")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "info":
        print(f"AFTERAI {__version__}")
        print("Status: early prototype")
        print("Training: opt-in; no model is downloaded or trained by this command.")
        return

    if args.command == "validate-data":
        errors = validate_jsonl(args.path)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            raise SystemExit(1)
        print(f"OK: {args.path} matches the AFTERAI chat JSONL format.")
        return

    raise SystemExit(f"Unknown command: {args.command}")
