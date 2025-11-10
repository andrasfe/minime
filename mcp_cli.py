"""
Command-line MCP client for interacting with the Digital Me MCP server.

Usage examples:

    # Store a chat summary
    python mcp_cli.py store-summary --summary "Talked about quantum computing."

    # Answer a question about the user
    python mcp_cli.py answer-question --question "Is Andras interested in quantum computing?"

    # Fetch the current digital me summary
    python mcp_cli.py get-digital-me
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.client import CallToolResult

DEFAULT_SERVER_URL = "http://127.0.0.1:8000/mcp"


def _extract_result(result: CallToolResult) -> Any:
    """Normalize CallToolResult into plain Python data."""
    if result.data is not None:
        return result.data
    if result.structured_content:
        return result.structured_content
    if result.content:
        texts: list[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text is not None:
                texts.append(text)
        if len(texts) == 1:
            return texts[0]
        if texts:
            return texts
    return None


def _normalize_url(url: str) -> str:
    if url.endswith("/"):
        url = url[:-1]
    if not url.endswith("/mcp"):
        url = f"{url}/mcp"
    return url


async def _call_tool(url: str, tool: str, arguments: dict[str, Any]) -> Any:
    target = _normalize_url(url)
    async with Client(target) as client:
        result = await client.call_tool(tool, arguments)
    return _extract_result(result)


async def _store_summary(args: argparse.Namespace) -> None:
    summary = _read_text_argument(args.summary, args.summary_file, "<summary>")
    payload = {"summary": summary}
    value = await _call_tool(args.url, "store_chat_summary", payload)
    _print_json(value)


async def _get_digital_me(args: argparse.Namespace) -> None:
    value = await _call_tool(args.url, "get_digital_me", {})
    _print_json(value)


async def _answer_question(args: argparse.Namespace) -> None:
    question = _read_text_argument(args.question, args.question_file, "<question>")
    payload = {"question": question}
    value = await _call_tool(args.url, "answer_question", payload)
    _print_json(value)


def _read_text_argument(
    value: str | None,
    file_path: str | None,
    placeholder: str,
) -> str:
    if value:
        return value
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise SystemExit(f"Error: {placeholder} text is required.")


def _print_json(value: Any) -> None:
    if value is None:
        print("null")
        return
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Command-line MCP client for the Digital Me MCP server."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_SERVER_URL,
        help=f"MCP server URL or base address (default: {DEFAULT_SERVER_URL}).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    store_parser = subparsers.add_parser(
        "store-summary", help="Store a chat summary via the MCP server."
    )
    store_parser.add_argument(
        "--summary",
        help="Summary text to store. If omitted, read from --summary-file or stdin.",
    )
    store_parser.add_argument(
        "--summary-file",
        help="Path to a file containing the summary text.",
    )
    store_parser.set_defaults(func=_store_summary)

    digital_me_parser = subparsers.add_parser(
        "get-digital-me", help="Retrieve the current digital me summary."
    )
    digital_me_parser.set_defaults(func=_get_digital_me)

    question_parser = subparsers.add_parser(
        "answer-question", help="Ask the MCP server an ad-hoc question about the user."
    )
    question_parser.add_argument(
        "--question",
        help="Question text to ask. If omitted, read from --question-file or stdin.",
    )
    question_parser.add_argument(
        "--question-file",
        help="Path to a file containing the question text.",
    )
    question_parser.set_defaults(func=_answer_question)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()


