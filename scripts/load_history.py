#!/usr/bin/env python3
"""
Load conversation history files into the Digital Me database.

This script processes conversation history files from OpenAI or Anthropic (Claude)
and loads them into the database using the MCP server's store_chat_summary tool.

Supported providers:
- openai: OpenAI ChatGPT conversation exports
- anthropic: Anthropic Claude conversation exports

Usage:
    python scripts/load_history.py --provider openai [--directory history/openAI]
    python scripts/load_history.py --provider anthropic [--directory history/anthropic]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Literal

from fastmcp import Client
from mcp import types

DEFAULT_SERVER_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_OPENAI_DIR = "history/openAI"
DEFAULT_ANTHROPIC_DIR = "history/anthropic"

ProviderType = Literal["openai", "anthropic"]


def _extract_result(result: list) -> Any:
    """Normalize tool result (list of content items) into plain Python data."""
    if not result:
        return None
    
    # Extract text from TextContent items
    texts: list[str] = []
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
    if url.endswith("/"):
        url = url[:-1]
    if not url.endswith("/mcp"):
        url = f"{url}/mcp"
    return url


async def _check_server(url: str) -> bool:
    """Check if the MCP server is reachable."""
    target = _normalize_url(url)
    try:
        async with Client(target) as client:
            # Try to list tools as a connectivity test
            await asyncio.wait_for(client.list_tools(), timeout=5.0)
            return True
    except Exception:
        return False


async def _store_summary(url: str, summary: str) -> dict:
    """Store a summary via the MCP server."""
    target = _normalize_url(url)
    try:
        async with Client(target) as client:
            result = await client.call_tool("store_chat_summary", {"summary": summary})
        return _extract_result(result)
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
            raise ConnectionError(
                f"Cannot connect to MCP server at {url}. "
                "Make sure the server is running:\n"
                "  python3 mcp_server.py\n"
                "Or with Docker:\n"
                "  docker-compose up -d"
            ) from e
        raise


def _validate_openai_conversation(data: dict) -> bool:
    """Validate OpenAI conversation structure."""
    if not isinstance(data, dict):
        return False
    # OpenAI format has "mapping" or "title" fields
    return "mapping" in data or "title" in data


def _validate_anthropic_conversation(data: dict) -> bool:
    """Validate Anthropic conversation structure."""
    if not isinstance(data, dict):
        return False
    # Anthropic format has uuid, chat_messages, and created_at
    required_fields = ["uuid", "chat_messages", "created_at"]
    return all(field in data for field in required_fields)


def _extract_openai_conversation(conversation_data: dict) -> str:
    """Extract conversation text from OpenAI conversation format."""
    messages = conversation_data.get("mapping", {})
    conversation_parts = []
    
    # Sort by creation time if available
    def get_create_time(item):
        message_data = item[1]
        message = message_data.get("message") if message_data else None
        if message and isinstance(message, dict):
            return message.get("create_time", 0) or 0
        return 0
    
    sorted_messages = sorted(messages.items(), key=get_create_time)
    
    for message_id, message_data in sorted_messages:
        message = message_data.get("message", {})
        if not message:
            continue
            
        author = message.get("author", {})
        if isinstance(author, dict):
            role = author.get("role", "")
        else:
            role = ""
        
        content = message.get("content", {})
        
        # Extract text from various content formats
        text_parts = []
        if isinstance(content, dict):
            if "parts" in content:
                for part in content["parts"]:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
            elif "text" in content:
                text_parts.append(str(content["text"]))
        elif isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
        
        # Add to conversation if we found text
        if text_parts:
            role_label = "User" if role == "user" else "Assistant" if role == "assistant" else role.title() if role else "Unknown"
            conversation_parts.append(f"{role_label}: {' '.join(text_parts)}")
    
    return "\n\n".join(conversation_parts)


def _extract_anthropic_conversation(conversation_data: dict) -> str:
    """Extract conversation text from Anthropic conversation format."""
    conversation_parts = []
    
    # Add conversation name/title if available
    name = conversation_data.get("name", "")
    if name:
        conversation_parts.append(f"Title: {name}")
    
    # Extract chat messages
    chat_messages = conversation_data.get("chat_messages", [])
    for message in chat_messages:
        sender = message.get("sender", "")
        text = message.get("text", "")
        
        if not text:
            # Try to extract from content array
            content = message.get("content", [])
            text_parts = []
            for content_item in content:
                if isinstance(content_item, dict) and content_item.get("type") == "text":
                    text_parts.append(content_item.get("text", ""))
            text = " ".join(text_parts)
        
        if text:
            role_label = "User" if sender == "human" else "Assistant" if sender == "assistant" else sender.title() if sender else "Unknown"
            conversation_parts.append(f"{role_label}: {text}")
    
    return "\n\n".join(conversation_parts)


def _process_file(file_path: Path, provider: ProviderType) -> list[str]:
    """Process a conversation history file and extract summaries."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle different file formats based on provider
        if isinstance(data, dict):
            # Try provider-specific format first
            if provider == "openai" and _validate_openai_conversation(data):
                conversation_text = _extract_openai_conversation(data)
                if conversation_text:
                    return [conversation_text]
            elif provider == "anthropic" and _validate_anthropic_conversation(data):
                conversation_text = _extract_anthropic_conversation(data)
                if conversation_text:
                    return [conversation_text]
            # Simple summary format (provider-agnostic)
            elif "summary" in data:
                return [data["summary"]]
            elif "text" in data:
                return [data["text"]]
        elif isinstance(data, list):
            # List of items - could be conversations or summaries
            summaries = []
            for item in data:
                if isinstance(item, str):
                    summaries.append(item)
                elif isinstance(item, dict):
                    # Check provider-specific formats
                    if provider == "openai" and _validate_openai_conversation(item):
                        conversation_text = _extract_openai_conversation(item)
                        if conversation_text:
                            summaries.append(conversation_text)
                    elif provider == "anthropic" and _validate_anthropic_conversation(item):
                        conversation_text = _extract_anthropic_conversation(item)
                        if conversation_text:
                            summaries.append(conversation_text)
                    # Otherwise treat as simple summary format
                    else:
                        summaries.append(item.get("summary", item.get("text", str(item))))
            return summaries
        elif isinstance(data, str):
            # Plain text file
            return [data.strip()]
        
        # Fallback: convert entire JSON to string
        return [json.dumps(data, indent=2)]
        
    except json.JSONDecodeError:
        # Try as plain text
        with open(file_path, "r", encoding="utf-8") as f:
            return [f.read().strip()]
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return []


async def _load_directory(directory: Path, url: str, provider: ProviderType, dry_run: bool = False):
    """Load all conversation files from a directory."""
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist", file=sys.stderr)
        return
    
    # Resolve to absolute path for consistent handling
    directory = directory.resolve()
    
    # Find all JSON and text files
    files = list(directory.rglob("*.json")) + list(directory.rglob("*.txt"))
    
    if not files:
        print(f"No JSON or text files found in {directory}", file=sys.stderr)
        return
    
    print(f"Found {len(files)} files to process (provider: {provider})")
    if dry_run:
        print("DRY RUN MODE - No data will be loaded")
    else:
        # Check if server is reachable
        print(f"Checking connection to MCP server at {url}...")
        if not await _check_server(url):
            print(f"\n❌ ERROR: Cannot connect to MCP server at {url}", file=sys.stderr)
            print("\nPlease start the MCP server first:", file=sys.stderr)
            print("  python3 mcp_server.py", file=sys.stderr)
            print("Or with Docker:", file=sys.stderr)
            print("  docker-compose up -d", file=sys.stderr)
            return
        print("✅ Server connection OK\n")
    
    loaded = 0
    failed = 0
    skipped = 0
    connection_errors = 0
    validation_errors = 0
    
    for file_path in files:
        # Use directory as base for relative path, fallback to absolute if needed
        try:
            rel_path = file_path.relative_to(directory)
            display_path = f"{directory.name}/{rel_path}" if directory.name != "." else str(rel_path)
        except ValueError:
            # Fallback to relative from cwd or absolute path
            try:
                display_path = str(file_path.relative_to(Path.cwd()))
            except ValueError:
                display_path = str(file_path)
        print(f"\nProcessing: {display_path}")
        
        summaries = _process_file(file_path, provider)
        
        if not summaries:
            print(f"  ⚠️  No content extracted or validation failed for {file_path.name}")
            validation_errors += 1
            continue
        
        for i, summary in enumerate(summaries):
            if len(summaries) > 1:
                print(f"  Summary {i+1}/{len(summaries)}")
            
            if dry_run:
                print(f"  Would load: {summary[:100]}...")
                loaded += 1
            else:
                try:
                    result = await _store_summary(url, summary)
                    status = result.get("status")
                    if status == "success":
                        if len(summaries) > 1:
                            print(f"  ✅ Loaded (ID: {result.get('chat_id')})")
                        else:
                            print(f"  ✅ Loaded successfully (ID: {result.get('chat_id')})")
                        loaded += 1
                    elif status == "skipped":
                        if len(summaries) > 1:
                            print(f"  ⏭️  Skipped (already exists, ID: {result.get('chat_id')})")
                        else:
                            print(f"  ⏭️  Skipped - already exists in database (ID: {result.get('chat_id')})")
                        skipped += 1
                    else:
                        print(f"  ❌ Failed: {result.get('message', 'Unknown error')}")
                        failed += 1
                except ConnectionError as e:
                    connection_errors += 1
                    if connection_errors == 1:
                        print(f"  ❌ Connection error: {e}")
                        print("  Stopping - please start the server and try again")
                        break
                    failed += 1
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    failed += 1
    
    print(f"\n{'='*50}")
    if connection_errors > 0:
        print(f"Summary: {loaded} loaded, {skipped} skipped, {failed} failed, {validation_errors} validation errors ({connection_errors} connection errors)")
        print("\n⚠️  Server connection was lost. Please restart the server and run again.")
    else:
        if validation_errors > 0:
            print(f"Summary: {loaded} loaded, {skipped} skipped, {failed} failed, {validation_errors} validation errors")
        elif skipped > 0:
            print(f"Summary: {loaded} loaded, {skipped} skipped, {failed} failed")
        else:
            print(f"Summary: {loaded} loaded, {failed} failed")
    if dry_run:
        print("(This was a dry run - no data was actually loaded)")


def main():
    parser = argparse.ArgumentParser(
        description="Load conversation history files into the Digital Me database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load OpenAI conversations
  python scripts/load_history.py --provider openai

  # Load Anthropic conversations
  python scripts/load_history.py --provider anthropic

  # Load from custom directory
  python scripts/load_history.py --provider openai --directory /path/to/conversations

  # Dry run to see what would be loaded
  python scripts/load_history.py --provider anthropic --dry-run
        """
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["openai", "anthropic"],
        help="Conversation provider (openai or anthropic)"
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=None,
        help=f"Directory containing history files (default: {DEFAULT_OPENAI_DIR} for openai, {DEFAULT_ANTHROPIC_DIR} for anthropic)"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_SERVER_URL,
        help=f"MCP server URL (default: {DEFAULT_SERVER_URL})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be loaded without actually loading"
    )
    
    args = parser.parse_args()
    
    # Set default directory based on provider if not specified
    if args.directory is None:
        if args.provider == "openai":
            args.directory = Path(DEFAULT_OPENAI_DIR)
        elif args.provider == "anthropic":
            args.directory = Path(DEFAULT_ANTHROPIC_DIR)
    
    asyncio.run(_load_directory(args.directory, args.url, args.provider, args.dry_run))


if __name__ == "__main__":
    main()

