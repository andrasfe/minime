"""
Unit tests for load_history.py script.

Tests validation and extraction functions for both OpenAI and Anthropic conversation formats.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from load_history import (
    _validate_openai_conversation,
    _validate_anthropic_conversation,
    _extract_openai_conversation,
    _extract_anthropic_conversation,
    _process_file,
)


class TestOpenAIValidation:
    """Test OpenAI conversation format validation."""

    def test_validate_openai_with_mapping(self):
        """Valid OpenAI conversation with mapping field."""
        data = {
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hello"]},
                        "create_time": 1234567890
                    }
                }
            }
        }
        assert _validate_openai_conversation(data) is True

    def test_validate_openai_with_title(self):
        """Valid OpenAI conversation with title field."""
        data = {
            "title": "Test Conversation",
            "create_time": 1234567890
        }
        assert _validate_openai_conversation(data) is True

    def test_validate_openai_invalid_not_dict(self):
        """Invalid OpenAI conversation - not a dict."""
        assert _validate_openai_conversation([]) is False
        assert _validate_openai_conversation("string") is False
        assert _validate_openai_conversation(None) is False

    def test_validate_openai_invalid_missing_fields(self):
        """Invalid OpenAI conversation - missing required fields."""
        data = {"random_field": "value"}
        assert _validate_openai_conversation(data) is False


class TestAnthropicValidation:
    """Test Anthropic conversation format validation."""

    def test_validate_anthropic_valid(self):
        """Valid Anthropic conversation with all required fields."""
        data = {
            "uuid": "test-uuid-123",
            "chat_messages": [],
            "created_at": "2024-08-08T14:31:43.657635Z"
        }
        assert _validate_anthropic_conversation(data) is True

    def test_validate_anthropic_with_messages(self):
        """Valid Anthropic conversation with messages."""
        data = {
            "uuid": "test-uuid-123",
            "chat_messages": [
                {
                    "uuid": "msg-uuid",
                    "text": "Hello",
                    "sender": "human",
                    "created_at": "2024-08-08T14:31:43.657635Z"
                }
            ],
            "created_at": "2024-08-08T14:31:43.657635Z"
        }
        assert _validate_anthropic_conversation(data) is True

    def test_validate_anthropic_invalid_not_dict(self):
        """Invalid Anthropic conversation - not a dict."""
        assert _validate_anthropic_conversation([]) is False
        assert _validate_anthropic_conversation("string") is False
        assert _validate_anthropic_conversation(None) is False

    def test_validate_anthropic_invalid_missing_fields(self):
        """Invalid Anthropic conversation - missing required fields."""
        # Missing uuid
        data1 = {"chat_messages": [], "created_at": "2024-08-08T14:31:43.657635Z"}
        assert _validate_anthropic_conversation(data1) is False

        # Missing chat_messages
        data2 = {"uuid": "test", "created_at": "2024-08-08T14:31:43.657635Z"}
        assert _validate_anthropic_conversation(data2) is False

        # Missing created_at
        data3 = {"uuid": "test", "chat_messages": []}
        assert _validate_anthropic_conversation(data3) is False


class TestOpenAIExtraction:
    """Test OpenAI conversation text extraction."""

    def test_extract_simple_conversation(self):
        """Extract simple OpenAI conversation."""
        data = {
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hello, how are you?"]},
                        "create_time": 1000
                    }
                },
                "msg2": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["I'm doing well, thank you!"]},
                        "create_time": 2000
                    }
                }
            }
        }
        result = _extract_openai_conversation(data)
        assert "User: Hello, how are you?" in result
        assert "Assistant: I'm doing well, thank you!" in result

    def test_extract_with_multiple_parts(self):
        """Extract OpenAI conversation with multiple content parts."""
        data = {
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Part 1", "Part 2", "Part 3"]},
                        "create_time": 1000
                    }
                }
            }
        }
        result = _extract_openai_conversation(data)
        assert "User: Part 1 Part 2 Part 3" in result

    def test_extract_with_text_content(self):
        """Extract OpenAI conversation with text field in content."""
        data = {
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"text": "Direct text content"},
                        "create_time": 1000
                    }
                }
            }
        }
        result = _extract_openai_conversation(data)
        assert "User: Direct text content" in result

    def test_extract_empty_messages(self):
        """Extract OpenAI conversation with empty messages."""
        data = {
            "mapping": {
                "msg1": {
                    "message": None
                },
                "msg2": {
                    "message": {}
                }
            }
        }
        result = _extract_openai_conversation(data)
        assert result == ""

    def test_extract_sorts_by_create_time(self):
        """Verify messages are sorted by create_time."""
        data = {
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Second message"]},
                        "create_time": 2000
                    }
                },
                "msg2": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["First message"]},
                        "create_time": 1000
                    }
                }
            }
        }
        result = _extract_openai_conversation(data)
        lines = result.split("\n\n")
        assert "First message" in lines[0]
        assert "Second message" in lines[1]


class TestAnthropicExtraction:
    """Test Anthropic conversation text extraction."""

    def test_extract_simple_conversation(self):
        """Extract simple Anthropic conversation."""
        data = {
            "uuid": "test-uuid",
            "name": "Test Conversation",
            "chat_messages": [
                {
                    "uuid": "msg1",
                    "text": "Hello, Claude!",
                    "sender": "human",
                    "created_at": "2024-08-08T14:31:43.657635Z"
                },
                {
                    "uuid": "msg2",
                    "text": "Hello! How can I help you today?",
                    "sender": "assistant",
                    "created_at": "2024-08-08T14:31:44.657635Z"
                }
            ],
            "created_at": "2024-08-08T14:31:43.657635Z"
        }
        result = _extract_anthropic_conversation(data)
        assert "Title: Test Conversation" in result
        assert "User: Hello, Claude!" in result
        assert "Assistant: Hello! How can I help you today?" in result

    def test_extract_without_title(self):
        """Extract Anthropic conversation without title."""
        data = {
            "uuid": "test-uuid",
            "name": "",
            "chat_messages": [
                {
                    "uuid": "msg1",
                    "text": "Test message",
                    "sender": "human",
                    "created_at": "2024-08-08T14:31:43.657635Z"
                }
            ],
            "created_at": "2024-08-08T14:31:43.657635Z"
        }
        result = _extract_anthropic_conversation(data)
        assert "Title:" not in result
        assert "User: Test message" in result

    def test_extract_with_content_array(self):
        """Extract Anthropic conversation with content array."""
        data = {
            "uuid": "test-uuid",
            "name": "",
            "chat_messages": [
                {
                    "uuid": "msg1",
                    "text": "",
                    "content": [
                        {
                            "type": "text",
                            "text": "Content from array"
                        }
                    ],
                    "sender": "human",
                    "created_at": "2024-08-08T14:31:43.657635Z"
                }
            ],
            "created_at": "2024-08-08T14:31:43.657635Z"
        }
        result = _extract_anthropic_conversation(data)
        assert "User: Content from array" in result

    def test_extract_empty_messages(self):
        """Extract Anthropic conversation with empty messages."""
        data = {
            "uuid": "test-uuid",
            "name": "",
            "chat_messages": [],
            "created_at": "2024-08-08T14:31:43.657635Z"
        }
        result = _extract_anthropic_conversation(data)
        assert result == ""


class TestProcessFile:
    """Test file processing with provider-specific formats."""

    def test_process_openai_file(self, tmp_path):
        """Process file with OpenAI format."""
        data = {
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Test message"]},
                        "create_time": 1000
                    }
                }
            }
        }
        file_path = tmp_path / "conversation.json"
        file_path.write_text(json.dumps(data))

        result = _process_file(file_path, "openai")
        assert len(result) == 1
        assert "User: Test message" in result[0]

    def test_process_anthropic_file(self, tmp_path):
        """Process file with Anthropic format."""
        data = {
            "uuid": "test-uuid",
            "name": "Test",
            "chat_messages": [
                {
                    "uuid": "msg1",
                    "text": "Test message",
                    "sender": "human",
                    "created_at": "2024-08-08T14:31:43.657635Z"
                }
            ],
            "created_at": "2024-08-08T14:31:43.657635Z"
        }
        file_path = tmp_path / "conversation.json"
        file_path.write_text(json.dumps(data))

        result = _process_file(file_path, "anthropic")
        assert len(result) == 1
        assert "User: Test message" in result[0]

    def test_process_list_of_conversations(self, tmp_path):
        """Process file with list of conversations."""
        data = [
            {
                "uuid": "test-uuid-1",
                "name": "Conv 1",
                "chat_messages": [
                    {
                        "uuid": "msg1",
                        "text": "Message 1",
                        "sender": "human",
                        "created_at": "2024-08-08T14:31:43.657635Z"
                    }
                ],
                "created_at": "2024-08-08T14:31:43.657635Z"
            },
            {
                "uuid": "test-uuid-2",
                "name": "Conv 2",
                "chat_messages": [
                    {
                        "uuid": "msg2",
                        "text": "Message 2",
                        "sender": "human",
                        "created_at": "2024-08-08T14:31:43.657635Z"
                    }
                ],
                "created_at": "2024-08-08T14:31:43.657635Z"
            }
        ]
        file_path = tmp_path / "conversations.json"
        file_path.write_text(json.dumps(data))

        result = _process_file(file_path, "anthropic")
        assert len(result) == 2
        assert "Message 1" in result[0]
        assert "Message 2" in result[1]

    def test_process_invalid_json(self, tmp_path):
        """Process file with invalid JSON (fallback to plain text)."""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("Not valid JSON")

        result = _process_file(file_path, "openai")
        assert len(result) == 1
        assert result[0] == "Not valid JSON"

    def test_process_wrong_provider_format(self, tmp_path):
        """Process file with wrong provider format."""
        # Anthropic data but openai provider
        data = {
            "uuid": "test-uuid",
            "name": "Test",
            "chat_messages": [],
            "created_at": "2024-08-08T14:31:43.657635Z"
        }
        file_path = tmp_path / "conversation.json"
        file_path.write_text(json.dumps(data))

        result = _process_file(file_path, "openai")
        # Should fallback to JSON dump since it doesn't match OpenAI format
        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

