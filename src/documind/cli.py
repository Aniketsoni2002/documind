"""Command-line interface for DocuMind.

Examples
--------
    documind ingest data/uploads/handbook.pdf
    documind ask "What is the refund policy?"
    documind reset
"""
from __future__ import annotations

import argparse
import sys

from documind.core.ingest import ingest_file
from documind.core.rag import answer_question
from documind.core.vectorstore import clear_collection


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="documind", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index a document.")
    p_ingest.add_argument("path", help="Path to a PDF/TXT/MD file.")

    p_ask = sub.add_parser("ask", help="Ask a question about indexed docs.")
    p_ask.add_argument("question", help="Your question, in quotes.")

    sub.add_parser("reset", help="Clear the vector store.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "ingest":
        n = ingest_file(args.path)
        print(f"Indexed {n} chunks from {args.path}")
    elif args.command == "ask":
        result = answer_question(args.question)
        print(result.text)
        if result.sources:
            print("\nSources: " + ", ".join(result.sources))
    elif args.command == "reset":
        clear_collection()
        print("Vector store cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
