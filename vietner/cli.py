"""Command line: ``vietner {entities,anonymize,serve-api}``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vietner.anonymize import STRATEGIES


def _read(arg: str | None) -> str:
    if arg is None or arg == "-":
        return sys.stdin.read()
    path = Path(arg)
    return path.read_text(encoding="utf-8") if path.is_file() else arg


def _load(args):
    from vietner.pipeline import DEFAULT_CHECKPOINT, VietNER

    return VietNER.from_checkpoint(args.checkpoint or DEFAULT_CHECKPOINT, device=args.device)


def cmd_entities(args) -> int:
    result = _load(args).analyze([_read(args.input)], ner=not args.rules_only)[0]
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0


def cmd_anonymize(args) -> int:
    result = _load(args).anonymize(_read(args.input), args.strategy, args.labels)
    if args.mapping:
        Path(args.mapping).write_text(json.dumps(result.mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.text)
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from vietner.api import create_app

    uvicorn.run(create_app(lambda: _load(args)), host=args.host, port=args.port)
    return 0


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", help="model checkpoint (default: models/best_epoch_16.pt)")
    common.add_argument("--device", default="auto")
    parser = argparse.ArgumentParser(prog="vietner", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("entities", parents=[common], help="print entities as JSON")
    p.add_argument("input", nargs="?", help="text, a file path, or - for stdin")
    p.add_argument("--rules-only", action="store_true")
    p.set_defaults(func=cmd_entities)

    p = sub.add_parser("anonymize", parents=[common], help="print anonymised text")
    p.add_argument("input", nargs="?", help="text, a file path, or - for stdin")
    p.add_argument("--strategy", choices=STRATEGIES, default="tag")
    p.add_argument("--labels", nargs="+", help="labels to replace (default: all personal-data labels)")
    p.add_argument("--mapping", help="write the placeholder -> original mapping (JSON) here")
    p.set_defaults(func=cmd_anonymize)

    p = sub.add_parser("serve-api", parents=[common], help="run the REST API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
