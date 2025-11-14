#!/usr/bin/env python3
"""
LangChain-powered CLI client for querying the Digital Me MCP server.

This client accepts a question, builds a LangChain Runnable pipeline that
formats the question with a prompt and invokes the MCP server's
``answer_question`` tool using FastMCP.  The result is printed in either JSON
or human-readable form.

Usage:
    python scripts/minime_client.py --question "Your question here"
"""

import argparse
import asyncio
import json
import sys
from typing import Any, Dict

from fastmcp import Client
from mcp import types
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda


DEFAULT_SERVER_URL = "http://127.0.0.1:8000/mcp"


def _extract_result(result: list) -> Any:
    """Normalize tool result (list of content items) into plain Python data."""
    if not result:
        return None
    
    # Extract text from TextContent items
    texts = []
    structured_data = None
    
    for item in result:
        if isinstance(item, types.TextContent):
            text = getattr(item, "text", None)
            if text:
                # Try to parse as JSON if it looks like structured data
                text_str = str(text).strip()
                if text_str.startswith("{") or text_str.startswith("["):
                    try:
                        import json
                        parsed = json.loads(text_str)
                        if isinstance(parsed, dict):
                            structured_data = parsed
                        else:
                            texts.append(text)
                    except (json.JSONDecodeError, ValueError):
                        texts.append(text)
                else:
                    texts.append(text)
        elif hasattr(item, "text"):
            # Fallback for other content types with text attribute
            texts.append(getattr(item, "text"))
    
    # Prefer structured data if found
    if structured_data:
        return structured_data
    
    # Return single text if only one, otherwise return list
    if len(texts) == 1:
        return texts[0]
    if texts:
        return texts
    
    return None


def _normalize_url(url: str) -> str:
    """Ensure the target URL ends with /mcp."""
    url = url.rstrip("/")
    if not url.endswith("/mcp"):
        url = f"{url}/mcp"
    return url


def _extract_question_from_prompt(prompt_value) -> str:
    """Return the human question string from a ChatPromptValue."""
    # LangChain 1.0.x ChatPromptTemplate.invoke returns a ChatPromptValue with messages
    # We expect the human message to contain the question
    messages = getattr(prompt_value, 'messages', [])
    for message in messages:
        if isinstance(message, HumanMessage):
            return message.content
    # Fallback to string representation if no human message found
    return str(prompt_value)


async def _call_mcp_answer(question: str, url: str) -> Dict[str, Any]:
    """Invoke the MCP server answer_question tool."""
    target = _normalize_url(url)
    try:
        async with Client(target) as client:
            result = await client.call_tool("answer_question", {"question": question})
        data = _extract_result(result)
        if isinstance(data, dict):
            return data
        return {"answer": data}
    except RuntimeError as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "failed to connect" in error_msg.lower():
            raise ConnectionError(
                f"Cannot connect to MCP server at {url}.\n\n"
                "Please make sure the MCP server is running:\n"
                "  python3 mcp_server.py\n"
                "Or with Docker:\n"
                "  docker-compose up -d\n\n"
                f"Server URL: {target}"
            ) from e
        raise


def build_chain(url: str):
    """Create a LangChain pipeline that asks the MCP server a question."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a command-line assistant that queries the Digital Me MCP server "
                "to retrieve answers about the user based on stored conversations.",
            ),
            ("human", "{question}"),
        ]
    )

    async def call_mcp_from_prompt(prompt_value) -> Dict[str, Any]:
        question = _extract_question_from_prompt(prompt_value)
        return await _call_mcp_answer(question, url)

    return prompt | RunnableLambda(call_mcp_from_prompt)


async def run_cli(question: str, url: str, output_format: str) -> None:
    chain = build_chain(url)
    result = await chain.ainvoke({"question": question})

    if output_format == "json":
        print(json.dumps({"question": question, "result": result}, indent=2, ensure_ascii=False))
    else:
        if isinstance(result, dict):
            answer = result.get("answer") or result.get("message") or result
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            context_sources = result.get("context_sources")
            if context_sources is not None:
                print(f"Context sources: {context_sources}")
        else:
            print(f"Question: {question}")
            print(f"Answer: {result}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions using LangChain and the Digital Me MCP server."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask the MCP server.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_SERVER_URL,
        help=f"MCP server base URL (default: {DEFAULT_SERVER_URL}).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_cli(args.question, args.url, args.format))
    except ConnectionError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

