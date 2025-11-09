"""
Unit tests for MCP server tools

These tests use mocks to test individual functions without database or LLM dependencies.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np
from datetime import datetime

import mcp_server


class TestGetLLM:
    """Test LLM initialization"""
    
    @patch.dict('os.environ', {'LLM_PROVIDER': 'openai', 'OPENAI_API_KEY': 'test-key'})
    @patch('mcp_server.ChatOpenAI')
    def test_get_llm_openai(self, mock_chat_openai):
        """Test OpenAI LLM initialization"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        result = mcp_server.get_llm()
        
        assert result == mock_llm
        mock_chat_openai.assert_called_once()
    
    @patch.dict('os.environ', {'LLM_PROVIDER': 'anthropic', 'ANTHROPIC_API_KEY': 'test-key'})
    @patch('mcp_server.ChatAnthropic')
    def test_get_llm_anthropic(self, mock_chat_anthropic):
        """Test Anthropic LLM initialization"""
        mock_llm = Mock()
        mock_chat_anthropic.return_value = mock_llm
        
        result = mcp_server.get_llm()
        
        assert result == mock_llm
        mock_chat_anthropic.assert_called_once()
    
    @patch.dict('os.environ', {'LLM_PROVIDER': 'openai'}, clear=True)
    def test_get_llm_missing_api_key(self):
        """Test LLM initialization fails without API key"""
        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            mcp_server.get_llm()


class TestGetEmbeddings:
    """Test embeddings initialization"""
    
    @patch.dict('os.environ', {'EMBEDDINGS_PROVIDER': 'openai', 'OPENAI_API_KEY': 'test-key'})
    @patch('mcp_server.OpenAIEmbeddings')
    def test_get_embeddings_openai(self, mock_embeddings):
        """Test OpenAI embeddings initialization"""
        mock_emb = Mock()
        mock_embeddings.return_value = mock_emb
        
        result = mcp_server.get_embeddings()
        
        assert result == mock_emb
        mock_embeddings.assert_called_once()
    
    @patch.dict('os.environ', {'EMBEDDINGS_PROVIDER': 'openai'}, clear=True)
    def test_get_embeddings_missing_api_key(self):
        """Test embeddings initialization fails without API key"""
        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
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
        mock_embeddings.embed_query.return_value = [0.1] * 1536
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
        mock_embeddings.embed_query.return_value = [0.1] * 1536
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
        mock_embeddings.embed_query.return_value = [0.1] * 1536
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

