from afterai.cli import build_parser


def test_validate_data_parser():
    args = build_parser().parse_args(["validate-data", "sample.jsonl"])
    assert args.command == "validate-data"
    assert args.path == "sample.jsonl"


def test_ask_parser():
    args = build_parser().parse_args(
        ["ask", "--model", "example/model", "--prompt", "Olá"]
    )
    assert args.command == "ask"
    assert args.model == "example/model"
    assert args.prompt == "Olá"
    assert args.device == "auto"
