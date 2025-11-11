"""
Unit tests for MCP server tools

These tests use mocks to test individual functions without database or LLM dependencies.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np
import json
import hashlib
from datetime import datetime

import mcp_server

EMBED_DIM = 4096


class TestGetLLM:
    """Test LLM initialization"""
    
    @patch.dict(
        'os.environ',
        {
            'OPENROUTER_API_KEY': 'test-key',
            'OPENROUTER_MODEL': 'moonshotai/kimi-k2-thinking',
            'OPENROUTER_URL': 'https://openrouter.ai/api/v1',
        },
        clear=True,
    )
    @patch('mcp_server.OpenRouterChat')
    def test_get_llm_openrouter(self, mock_openrouter_chat):
        """Test OpenRouter LLM initialization"""
        mock_llm = Mock()
        mock_openrouter_chat.return_value = mock_llm
        
        result = mcp_server.get_llm()
        
        assert result == mock_llm
        mock_openrouter_chat.assert_called_once()
    
    @patch.dict('os.environ', {'OPENROUTER_MODEL': 'test-model'}, clear=True)
    def test_get_llm_missing_api_key(self):
        """Test LLM initialization fails without API key"""
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY not found"):
            mcp_server.get_llm()


class TestGetEmbeddings:
    """Test embeddings initialization"""
    
    @patch.dict(
        'os.environ',
        {
            'EMBEDDING_PROVIDER': 'cohere-v2',
            'COHERE_API_KEY': 'test-key',
            'EMBEDDING_MODEL': 'embed-english-v2.0',
        },
        clear=True,
    )
    @patch('mcp_server.CohereEmbeddingsClient')
    def test_get_embeddings_cohere(self, mock_embeddings_cls):
        """Test Cohere embeddings initialization"""
        mock_emb = Mock()
        mock_embeddings_cls.return_value = mock_emb
        
        result = mcp_server.get_embeddings()
        
        assert result == mock_emb
        mock_embeddings_cls.assert_called_once()
    
    @patch.dict('os.environ', {'EMBEDDING_PROVIDER': 'cohere'}, clear=True)
    def test_get_embeddings_missing_api_key(self):
        """Test embeddings initialization fails without API key"""
        with pytest.raises(ValueError, match="COHERE_API_KEY not found"):
            mcp_server.get_embeddings()


class TestStoreChatSummary:
    """Unit tests for store_chat_summary tool"""
    
    @patch('mcp_server.init_database')
    @patch('mcp_server.get_llm')
    @patch('mcp_server.get_embeddings')
    @patch('mcp_server.get_db_connection')
    @patch('mcp_server.update_digital_me')
    def test_store_chat_summary_success(
        self, mock_update_digital_me, mock_get_db_conn,
        mock_get_embeddings, mock_get_llm, mock_init_db
    ):
        """Test successful chat summary storage"""
        # Setup mocks
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = "Processed summary for RAG"
        mock_llm.invoke.return_value = mock_llm_response
        mock_get_llm.return_value = mock_llm
        
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * EMBED_DIM
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_result = {'id': 1, 'created_at': datetime.now()}
        mock_cursor.fetchone.return_value = mock_result
        mock_get_db_conn.return_value = mock_conn
        
        # Execute - access the underlying function from the tool
        result = mcp_server.store_chat_summary.fn("Test summary")
        
        # Assertions
        assert result["status"] == "success"
        assert result["chat_id"] == 1
        assert "created_at" in result
        mock_init_db.assert_called_once()
        mock_get_llm.assert_called()
        mock_get_embeddings.assert_called()
        mock_update_digital_me.assert_called_once()
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called()
    
    @patch('mcp_server.init_database')
    @patch('mcp_server.get_llm')
    def test_store_chat_summary_error(self, mock_get_llm, mock_init_db):
        """Test error handling in store_chat_summary"""
        mock_get_llm.side_effect = Exception("LLM error")
        
        result = mcp_server.store_chat_summary.fn("Test summary")
        
        assert result["status"] == "error"
        assert "error" in result["message"].lower()
    
    @patch('mcp_server.init_database')
    @patch('mcp_server.get_llm')
    @patch('mcp_server.get_embeddings')
    @patch('mcp_server.get_db_connection')
    def test_store_chat_summary_duplicate_skipped(
        self, mock_get_db_conn, mock_get_embeddings, mock_get_llm, mock_init_db
    ):
        """Test that duplicate summaries are skipped"""
        # Setup mocks for LLM and embeddings (required before duplicate check)
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = "Processed summary"
        mock_llm.invoke.return_value = mock_llm_response
        mock_get_llm.return_value = mock_llm
        
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * EMBED_DIM
        mock_get_embeddings.return_value = mock_embeddings
        
        # Setup database mocks
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock: existing summary found (duplicate)
        existing_result = {'id': 42}
        mock_cursor.fetchone.return_value = existing_result
        mock_get_db_conn.return_value = mock_conn
        
        test_summary = "Test summary text"
        result = mcp_server.store_chat_summary.fn(test_summary)
        
        # Assertions
        assert result["status"] == "skipped"
        assert result["chat_id"] == 42
        assert result["reason"] == "duplicate"
        assert "already exists" in result["message"].lower()
        
        # Verify duplicate check query was executed
        calls = mock_cursor.execute.call_args_list
        duplicate_check_call = None
        for call_args in calls:
            if call_args and len(call_args[0]) > 0:
                sql = call_args[0][0]
                if "summary_hash" in sql and "SELECT" in sql:
                    duplicate_check_call = call_args
                    break
        
        assert duplicate_check_call is not None, "Duplicate check query should be executed"
        
        # Verify hash was computed correctly
        expected_hash = hashlib.sha256(test_summary.encode('utf-8')).hexdigest()
        # Check that the hash was used in the query
        assert expected_hash in str(duplicate_check_call)
        
        # Verify no INSERT was called (since it's a duplicate)
        insert_calls = [c for c in calls if c and len(c[0]) > 0 and "INSERT" in c[0][0]]
        assert len(insert_calls) == 0, "INSERT should not be called for duplicates"
        
        # Verify update_digital_me was NOT called for duplicates
        # (We need to check this by ensuring it's not in the call stack)
    
    @patch('mcp_server.init_database')
    @patch('mcp_server.get_llm')
    @patch('mcp_server.get_embeddings')
    @patch('mcp_server.get_db_connection')
    @patch('mcp_server.update_digital_me')
    def test_store_chat_summary_stores_hash_in_metadata(
        self, mock_update_digital_me, mock_get_db_conn,
        mock_get_embeddings, mock_get_llm, mock_init_db
    ):
        """Test that summary hash is stored in metadata"""
        # Setup mocks
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = "Processed summary"
        mock_llm.invoke.return_value = mock_llm_response
        mock_get_llm.return_value = mock_llm
        
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * EMBED_DIM
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock: no existing summary (not duplicate)
        mock_cursor.fetchone.side_effect = [
            None,  # First call: duplicate check returns None
            {'id': 1, 'created_at': datetime.now()}  # Second call: INSERT returns result
        ]
        mock_get_db_conn.return_value = mock_conn
        
        test_summary = "Unique test summary"
        result = mcp_server.store_chat_summary.fn(test_summary)
        
        # Assertions
        assert result["status"] == "success"
        
        # Verify INSERT was called with hash in metadata
        insert_calls = [c for c in mock_cursor.execute.call_args_list 
                       if c and len(c[0]) > 0 and "INSERT" in c[0][0]]
        assert len(insert_calls) > 0, "INSERT should be called for new summaries"
        
        # Extract metadata from INSERT call
        insert_call = insert_calls[0]
        # The metadata is the third parameter (index 2 in args[1])
        metadata_json = insert_call[0][1][2]  # args[1] is the tuple of values
        metadata = json.loads(metadata_json)
        
        # Verify hash is in metadata
        assert "summary_hash" in metadata
        assert "original_summary" in metadata
        assert metadata["original_summary"] == test_summary
        
        # Verify hash is correct
        expected_hash = hashlib.sha256(test_summary.encode('utf-8')).hexdigest()
        assert metadata["summary_hash"] == expected_hash
    
    @patch('mcp_server.init_database')
    @patch('mcp_server.get_llm')
    @patch('mcp_server.get_embeddings')
    @patch('mcp_server.get_db_connection')
    @patch('mcp_server.update_digital_me')
    def test_store_chat_summary_hash_computation(
        self, mock_update_digital_me, mock_get_db_conn,
        mock_get_embeddings, mock_get_llm, mock_init_db
    ):
        """Test that hash is computed correctly for different summaries"""
        # Setup mocks
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = "Processed"
        mock_llm.invoke.return_value = mock_llm_response
        mock_get_llm.return_value = mock_llm
        
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * EMBED_DIM
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [
            None,  # No duplicate
            {'id': 1, 'created_at': datetime.now()}
        ]
        mock_get_db_conn.return_value = mock_conn
        
        # Test with different summaries produce different hashes
        summary1 = "First summary"
        summary2 = "Second summary"
        
        hash1 = hashlib.sha256(summary1.encode('utf-8')).hexdigest()
        hash2 = hashlib.sha256(summary2.encode('utf-8')).hexdigest()
        
        assert hash1 != hash2, "Different summaries should produce different hashes"
        
        # Verify the hash is used in duplicate check
        result1 = mcp_server.store_chat_summary.fn(summary1)
        assert result1["status"] == "success"
        
        # Check that the hash was used in the duplicate check query
        duplicate_check = mock_cursor.execute.call_args_list[0]
        assert hash1 in str(duplicate_check), "Hash should be used in duplicate check"


class TestGetDigitalMe:
    """Unit tests for get_digital_me tool"""
    
    @patch('mcp_server.init_database')
    @patch('mcp_server.get_db_connection')
    def test_get_digital_me_success(self, mock_get_db_conn, mock_init_db):
        """Test successful retrieval of digital me"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_result = {
            'summary_text': 'Test summary',
            'updated_at': datetime.now()
        }
        mock_cursor.fetchone.return_value = mock_result
        mock_get_db_conn.return_value = mock_conn
        
        result = mcp_server.get_digital_me.fn()
        
        assert result["status"] == "success"
        assert result["summary"] == "Test summary"
        assert "updated_at" in result
        mock_init_db.assert_called_once()
    
    @patch('mcp_server.init_database')
    @patch('mcp_server.get_db_connection')
    def test_get_digital_me_empty(self, mock_get_db_conn, mock_init_db):
        """Test retrieval when digital me is empty"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        mock_get_db_conn.return_value = mock_conn
        
        result = mcp_server.get_digital_me.fn()
        
        assert result["status"] == "success"
        assert result["summary"] == ""
        assert result.get("message") == "Digital me record is empty"
    
    @patch('mcp_server.init_database')
    def test_get_digital_me_error(self, mock_init_db):
        """Test error handling in get_digital_me"""
        mock_init_db.side_effect = Exception("Database error")
        
        result = mcp_server.get_digital_me.fn()
        
        assert result["status"] == "error"
        assert "error" in result["message"].lower()


class TestAnswerQuestion:
    """Unit tests for answer_question tool"""
    
    @patch('mcp_server.init_database')
    @patch('mcp_server.get_llm')
    @patch('mcp_server.get_embeddings')
    @patch('mcp_server.get_db_connection')
    def test_answer_question_success(
        self, mock_get_db_conn, mock_get_embeddings,
        mock_get_llm, mock_init_db
    ):
        """Test successful question answering"""
        # Setup mocks
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * EMBED_DIM
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = "Yes, Andras is interested in quantum computing."
        mock_llm.invoke.return_value = mock_llm_response
        mock_get_llm.return_value = mock_llm
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            {'summary_text': 'Relevant summary 1', 'metadata': {}, 'similarity': 0.8}
        ]
        mock_cursor.fetchone.return_value = {'summary_text': 'Digital me summary'}
        mock_get_db_conn.return_value = mock_conn
        
        # Execute - access the underlying function from the tool
        result = mcp_server.answer_question.fn("Is Andras interested in quantum computing?")
        
        # Assertions
        assert result["status"] == "success"
        assert "answer" in result
        assert result["answer"] == "Yes, Andras is interested in quantum computing."
        assert result["context_sources"] > 0
        mock_init_db.assert_called_once()
        mock_get_embeddings.assert_called()
        mock_get_llm.assert_called()
    
    @patch('mcp_server.init_database')
    def test_answer_question_error(self, mock_init_db):
        """Test error handling in answer_question"""
        mock_init_db.side_effect = Exception("Database error")
        
        result = mcp_server.answer_question.fn("Test question?")
        
        assert result["status"] == "error"
        assert "error" in result["message"].lower()


class TestUpdateDigitalMe:
    """Unit tests for update_digital_me function"""
    
    @patch('mcp_server.get_llm')
    @patch('mcp_server.get_embeddings')
    @patch('mcp_server.get_db_connection')
    def test_update_digital_me_success(
        self, mock_get_db_conn, mock_get_embeddings, mock_get_llm
    ):
        """Test successful digital me update"""
        # Setup mocks
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = "Updated summary"
        mock_llm.invoke.return_value = mock_llm_response
        mock_get_llm.return_value = mock_llm
        
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * EMBED_DIM
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {'summary_text': 'Current summary'}
        mock_get_db_conn.return_value = mock_conn
        
        # Execute
        mcp_server.update_digital_me("New information")
        
        # Assertions
        mock_get_llm.assert_called()
        mock_get_embeddings.assert_called()
        assert mock_cursor.execute.call_count >= 2  # SELECT and UPDATE
        mock_conn.commit.assert_called()
    
    @patch('mcp_server.get_llm')
    @patch('mcp_server.get_db_connection')
    def test_update_digital_me_handles_error(self, mock_get_db_conn, mock_get_llm):
        """Test that update_digital_me handles errors gracefully"""
        mock_get_llm.side_effect = Exception("LLM error")
        
        # Should not raise exception, just log warning
        try:
            mcp_server.update_digital_me("New information")
        except Exception:
            pytest.fail("update_digital_me should handle errors gracefully")

