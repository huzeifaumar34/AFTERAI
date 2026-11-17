import json

from afterai.data import validate_jsonl


def test_valid_chat_jsonl(tmp_path):
    path = tmp_path / "valid.jsonl"
    record = {
        "messages": [
            {"role": "user", "content": "Olá"},
            {"role": "assistant", "content": "Olá!"},
        ]
    }
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    assert validate_jsonl(path) == []


def test_rejects_record_without_assistant(tmp_path):
    path = tmp_path / "invalid.jsonl"
    record = {"messages": [{"role": "user", "content": "Olá"}]}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    errors = validate_jsonl(path)
    assert any("needs an assistant message" in error for error in errors)


def test_rejects_invalid_json(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")

    errors = validate_jsonl(path)
    assert any("invalid JSON" in error for error in errors)


def test_reports_missing_file(tmp_path):
    errors = validate_jsonl(tmp_path / "missing.jsonl")
    assert errors and "does not exist" in errors[0]
