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
DEFAULT_LMSTUDIO_DIR = "history/lmstudio/conversations"

ProviderType = Literal["openai", "anthropic", "lmstudio"]


def _extract_result(result) -> Any:
    """Normalize tool result (list of content items) into plain Python data."""
    # Handle case where result might not be a list (legacy CallToolResult support)
    if not isinstance(result, list):
        # Check for CallToolResult-like objects first
        result_type_name = type(result).__name__
        if 'CallToolResult' in result_type_name or 'ToolResult' in result_type_name:
            # Handle CallToolResult object
            if hasattr(result, 'data') and result.data is not None:
                return result.data
            if hasattr(result, 'structured_content') and result.structured_content:
                return result.structured_content
            if hasattr(result, 'content'):
                # content might be a list or iterable
                content = result.content
                if isinstance(content, list):
                    result = content
                elif content:
                    # Try to make it a list, but be careful
                    try:
                        if hasattr(content, '__iter__') and not isinstance(content, (str, bytes)):
                            result = list(content)
                        else:
                            result = [content]
                    except (TypeError, ValueError):
                        return str(content) if content else None
                else:
                    return None
            else:
                # No content attribute, return string representation
                return str(result) if result else None
        else:
            # Not a CallToolResult, try other extraction methods
            if hasattr(result, 'data') and result.data is not None:
                return result.data
            if hasattr(result, 'structured_content') and result.structured_content:
                return result.structured_content
            # Try to convert to list only if it's actually iterable
            try:
                # Only try to iterate if it's actually iterable
                if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                    result = list(result)
                else:
                    return str(result) if result else None
            except (TypeError, ValueError) as e:
                # If we can't iterate, return string representation
                return str(result) if result else None
    
    # At this point, result should be a list
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


async def _store_summary(url: str, summary: str, original_date: str | None = None, use_local: bool = False) -> dict:
    """Store a summary via the MCP server.
    
    Args:
        url: MCP server URL
        summary: Conversation summary text
        original_date: ISO format date string of original conversation (optional)
        use_local: If True, use local LLM for processing (for sensitive data)
    """
    target = _normalize_url(url)
    try:
        async with Client(target) as client:
            # Pass original_date and use_local if applicable
            tool_args = {"summary": summary}
            if original_date:
                tool_args["original_date"] = original_date
            if use_local:
                tool_args["use_local"] = True
            result = await client.call_tool("store_chat_summary", tool_args)
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


def _extract_openai_conversation(conversation_data: dict) -> tuple[str, str | None]:
    """Extract conversation text and original date from OpenAI conversation format.
    
    Returns:
        Tuple of (conversation_text, original_date_iso) where date is ISO format string or None
    """
    messages = conversation_data.get("mapping", {})
    conversation_parts = []
    earliest_time = None
    
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
        
        # Extract creation time for date tracking
        create_time = message.get("create_time")
        if create_time and (earliest_time is None or create_time < earliest_time):
            earliest_time = create_time
            
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
    
    conversation_text = "\n\n".join(conversation_parts)
    
    # Convert timestamp to ISO format if available
    original_date = None
    if earliest_time:
        try:
            from datetime import datetime
            # OpenAI timestamps are Unix timestamps
            dt = datetime.fromtimestamp(earliest_time)
            original_date = dt.isoformat()
        except (ValueError, TypeError, OSError):
            pass
    
    return conversation_text, original_date


def _extract_anthropic_conversation(conversation_data: dict) -> tuple[str, str | None]:
    """Extract conversation text and original date from Anthropic conversation format.
    
    Returns:
        Tuple of (conversation_text, original_date_iso) where date is ISO format string or None
    """
    conversation_parts = []
    
    # Extract original conversation date
    original_date = None
    created_at = conversation_data.get("created_at")
    if created_at:
        try:
            from datetime import datetime
            # Anthropic dates might be ISO strings or timestamps
            if isinstance(created_at, (int, float)):
                dt = datetime.fromtimestamp(created_at)
            elif isinstance(created_at, str):
                # Try parsing ISO format
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                dt = None
            if dt:
                original_date = dt.isoformat()
        except (ValueError, TypeError, AttributeError):
            pass
    
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
    
    conversation_text = "\n\n".join(conversation_parts)
    return conversation_text, original_date


def _validate_lmstudio_conversation(data: dict) -> bool:
    """Validate LM Studio conversation structure."""
    if not isinstance(data, dict):
        return False
    # LM Studio format has "messages" array and typically "name" and "createdAt"
    return "messages" in data and isinstance(data.get("messages"), list)


def _extract_lmstudio_conversation(conversation_data: dict) -> tuple[str, str | None]:
    """Extract conversation text and original date from LM Studio conversation format.
    
    Returns:
        Tuple of (conversation_text, original_date_iso) where date is ISO format string or None
    """
    conversation_parts = []
    
    # Extract original conversation date from createdAt (milliseconds timestamp)
    original_date = None
    created_at = conversation_data.get("createdAt")
    if created_at:
        try:
            from datetime import datetime
            # LM Studio uses milliseconds timestamp
            if isinstance(created_at, (int, float)):
                dt = datetime.fromtimestamp(created_at / 1000)  # Convert ms to seconds
                original_date = dt.isoformat()
        except (ValueError, TypeError, OSError):
            pass
    
    # Add conversation name/title if available
    name = conversation_data.get("name", "")
    if name:
        conversation_parts.append(f"Title: {name}")
    
    # Add system prompt if present
    system_prompt = conversation_data.get("systemPrompt", "")
    if system_prompt:
        conversation_parts.append(f"System: {system_prompt}")
    
    # Extract messages
    messages = conversation_data.get("messages", [])
    for message in messages:
        # LM Studio uses "versions" array for message history (regenerations/edits)
        versions = message.get("versions", [])
        if not versions:
            continue
        
        # Use the currently selected version, or first version
        selected_idx = message.get("currentlySelected", 0)
        if selected_idx < len(versions):
            version = versions[selected_idx]
        else:
            version = versions[0]
        
        role = version.get("role", "unknown")
        content = version.get("content", [])
        
        # Extract text from content array
        text_parts = []
        if isinstance(content, list):
            for content_item in content:
                if isinstance(content_item, dict) and content_item.get("type") == "text":
                    text_parts.append(content_item.get("text", ""))
                elif isinstance(content_item, str):
                    text_parts.append(content_item)
        elif isinstance(content, str):
            text_parts.append(content)
        
        text = " ".join(text_parts).strip()
        if text:
            role_label = "User" if role == "user" else "Assistant" if role == "assistant" else role.title()
            conversation_parts.append(f"{role_label}: {text}")
    
    conversation_text = "\n\n".join(conversation_parts)
    return conversation_text, original_date


def _process_file(file_path: Path, provider: ProviderType) -> list[tuple[str, str | None]]:
    """Process a conversation history file and extract summaries with dates.
    
    Returns:
        List of tuples: (conversation_text, original_date_iso) where date can be None
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle different file formats based on provider
        if isinstance(data, dict):
            # Try provider-specific format first
            if provider == "openai" and _validate_openai_conversation(data):
                conversation_text, original_date = _extract_openai_conversation(data)
                if conversation_text:
                    return [(conversation_text, original_date)]
            elif provider == "anthropic" and _validate_anthropic_conversation(data):
                conversation_text, original_date = _extract_anthropic_conversation(data)
                if conversation_text:
                    return [(conversation_text, original_date)]
            elif provider == "lmstudio" and _validate_lmstudio_conversation(data):
                conversation_text, original_date = _extract_lmstudio_conversation(data)
                if conversation_text:
                    return [(conversation_text, original_date)]
            # Simple summary format (provider-agnostic) - no date available
            elif "summary" in data:
                return [(data["summary"], None)]
            elif "text" in data:
                return [(data["text"], None)]
        elif isinstance(data, list):
            # List of items - could be conversations or summaries
            summaries = []
            for item in data:
                if isinstance(item, str):
                    summaries.append((item, None))
                elif isinstance(item, dict):
                    # Check provider-specific formats
                    if provider == "openai" and _validate_openai_conversation(item):
                        conversation_text, original_date = _extract_openai_conversation(item)
                        if conversation_text:
                            summaries.append((conversation_text, original_date))
                    elif provider == "anthropic" and _validate_anthropic_conversation(item):
                        conversation_text, original_date = _extract_anthropic_conversation(item)
                        if conversation_text:
                            summaries.append((conversation_text, original_date))
                    elif provider == "lmstudio" and _validate_lmstudio_conversation(item):
                        conversation_text, original_date = _extract_lmstudio_conversation(item)
                        if conversation_text:
                            summaries.append((conversation_text, original_date))
                    # Otherwise treat as simple summary format - no date available
                    else:
                        summaries.append((item.get("summary", item.get("text", str(item))), None))
            return summaries
        elif isinstance(data, str):
            # Plain text file - no date available
            return [(data.strip(), None)]
        
        # Fallback: convert entire JSON to string - no date available
        return [(json.dumps(data, indent=2), None)]
        
    except json.JSONDecodeError:
        # Try as plain text - no date available
        with open(file_path, "r", encoding="utf-8") as f:
            return [(f.read().strip(), None)]
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return []


async def _load_directory(directory: Path, url: str, provider: ProviderType, dry_run: bool = False, use_local: bool = False):
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
    if use_local:
        print("🔒 SENSITIVE MODE - Using local LLM (no cloud API calls)")
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
        
        summaries_with_dates = _process_file(file_path, provider)
        
        if not summaries_with_dates:
            print(f"  ⚠️  No content extracted or validation failed for {file_path.name}")
            validation_errors += 1
            continue
        
        for i, (summary, original_date) in enumerate(summaries_with_dates):
            if len(summaries_with_dates) > 1:
                print(f"  Summary {i+1}/{len(summaries_with_dates)}")
            
            if dry_run:
                date_info = f" (date: {original_date})" if original_date else ""
                print(f"  Would load: {summary[:100]}...{date_info}")
                loaded += 1
            else:
                try:
                    result = await _store_summary(url, summary, original_date, use_local)
                    status = result.get("status")
                    if status == "success":
                        date_info = f" (original date: {original_date})" if original_date else ""
                        if len(summaries_with_dates) > 1:
                            print(f"  ✅ Loaded (ID: {result.get('chat_id')}){date_info}")
                        else:
                            print(f"  ✅ Loaded successfully (ID: {result.get('chat_id')}){date_info}")
                        loaded += 1
                    elif status == "skipped":
                        if len(summaries_with_dates) > 1:
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

  # Load LM Studio conversations (always uses local LLM)
  python scripts/load_history.py --provider lmstudio

  # Load from custom directory
  python scripts/load_history.py --provider openai --directory /path/to/conversations

  # Dry run to see what would be loaded
  python scripts/load_history.py --provider anthropic --dry-run

  # Load sensitive data using local LLM (no cloud API calls)
  python scripts/load_history.py --provider openai --sensitive
        """
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["openai", "anthropic", "lmstudio"],
        help="Conversation provider (openai, anthropic, or lmstudio)"
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=None,
        help=f"Directory containing history files (default: {DEFAULT_OPENAI_DIR} for openai, {DEFAULT_ANTHROPIC_DIR} for anthropic, {DEFAULT_LMSTUDIO_DIR} for lmstudio)"
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
    parser.add_argument(
        "--sensitive",
        action="store_true",
        help="Use local LLM for sensitive data (requires LOCAL_LLM_URL in .env)"
    )
    
    args = parser.parse_args()
    
    # Set default directory based on provider if not specified
    if args.directory is None:
        if args.provider == "openai":
            args.directory = Path(DEFAULT_OPENAI_DIR)
        elif args.provider == "anthropic":
            args.directory = Path(DEFAULT_ANTHROPIC_DIR)
        elif args.provider == "lmstudio":
            args.directory = Path(DEFAULT_LMSTUDIO_DIR)
    
    # LM Studio conversations are always sensitive - force local LLM
    use_local = args.sensitive
    if args.provider == "lmstudio":
        use_local = True
        if not args.sensitive:
            print("ℹ️  LM Studio conversations are always processed with local LLM (sensitive data)")
    
    asyncio.run(_load_directory(args.directory, args.url, args.provider, args.dry_run, use_local))


if __name__ == "__main__":
    main()

