"""Dataset format checks for AFTERAI supervised fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path

_ALLOWED_ROLES = {"system", "user", "assistant"}


def validate_jsonl(path: str | Path) -> list[str]:
    """Return validation errors for a JSONL chat dataset."""
    file_path = Path(path)
    errors: list[str] = []
    record_count = 0

    if not file_path.exists():
        return [f"{file_path}: file does not exist"]
    if not file_path.is_file():
        return [f"{file_path}: path is not a file"]

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                record_count += 1
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    errors.append(
                        f"{file_path}:{line_number}: invalid JSON ({exc.msg})"
                    )
                    continue

                if not isinstance(record, dict):
                    errors.append(
                        f"{file_path}:{line_number}: record must be a JSON object"
                    )
                    continue

                messages = record.get("messages")
                if not isinstance(messages, list) or not messages:
                    errors.append(
                        f"{file_path}:{line_number}: messages must be a non-empty list"
                    )
                    continue

                seen_roles: set[str] = set()
                for index, message in enumerate(messages):
                    location = f"{file_path}:{line_number}: message {index}"
                    if not isinstance(message, dict):
                        errors.append(f"{location} must be an object")
                        continue

                    role = message.get("role")
                    content = message.get("content")
                    if not isinstance(role, str) or role not in _ALLOWED_ROLES:
                        errors.append(
                            f"{location} has an unsupported role; use system, user, or assistant"
                        )
                    else:
                        seen_roles.add(role)

                    if not isinstance(content, str) or not content.strip():
                        errors.append(f"{location} must have non-empty string content")

                if "user" not in seen_roles:
                    errors.append(f"{file_path}:{line_number}: record needs a user message")
                if "assistant" not in seen_roles:
                    errors.append(
                        f"{file_path}:{line_number}: record needs an assistant message"
                    )
    except OSError as exc:
        return [f"{file_path}: could not read file ({exc})"]

    if record_count == 0:
        errors.append(f"{file_path}: dataset contains no records")
    return errors
