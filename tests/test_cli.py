from afterai.cli import build_parser


def test_validate_data_parser():
    args = build_parser().parse_args(["validate-data", "sample.jsonl"])
    assert args.command == "validate-data"
    assert args.path == "sample.jsonl"
