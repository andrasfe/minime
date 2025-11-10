"""
Integration tests for MCP server

These tests use a real test database to verify end-to-end functionality.
Requires TEST_DATABASE_URL environment variable to be set.
"""

import pytest
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import numpy as np
from unittest.mock import patch, Mock

import mcp_server

EMBED_DIM = 4096


@pytest.mark.integration
class TestDatabaseInitialization:
    """Test database schema initialization"""
    
    def test_init_database_creates_tables(self, clean_test_db, test_db_url):
        """Test that init_database creates required tables"""
        with patch.dict('os.environ', {'DATABASE_URL': test_db_url}):
            mcp_server.init_database()
        
        conn = clean_test_db
        with conn.cursor() as cur:
            # Check chat_summaries table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'chat_summaries'
                );
            """)
            assert cur.fetchone()[0] is True
            
            # Check digital_me table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'digital_me'
                );
            """)
            assert cur.fetchone()[0] is True
            
            # Check pgvector extension is enabled
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector');")
            assert cur.fetchone()[0] is True
    
    def test_init_database_creates_indexes(self, clean_test_db, test_db_url):
        """Test that init_database creates vector indexes"""
        with patch.dict('os.environ', {'DATABASE_URL': test_db_url}):
            mcp_server.init_database()
        
        conn = clean_test_db
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM pg_indexes 
                    WHERE indexname = 'chat_summaries_embedding_idx'
                );
            """
            )
            exists = cur.fetchone()[0]
            embedding_dim = int(os.getenv("EMBEDDING_DIMENSION", "0") or "0")
            if embedding_dim and embedding_dim > 2000:
                assert exists is False
            else:
                assert exists is True
    
    def test_init_database_initializes_digital_me(self, clean_test_db, test_db_url):
        """Test that init_database creates initial digital_me record"""
        with patch.dict('os.environ', {'DATABASE_URL': test_db_url}):
            mcp_server.init_database()
        
        conn = clean_test_db
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM digital_me WHERE id = 1;")
            result = cur.fetchone()
            assert result is not None
            assert result['id'] == 1
            assert result['summary_text'] == ''


@pytest.mark.integration
class TestStoreChatSummaryIntegration:
    """Integration tests for store_chat_summary"""
    
    @patch('mcp_server.get_llm')
    @patch('mcp_server.get_embeddings')
    def test_store_chat_summary_full_flow(
        self, mock_get_embeddings, mock_get_llm,
        clean_test_db, test_db_url, sample_chat_summary
    ):
        """Test complete flow of storing a chat summary"""
        # Setup mocks
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = "Processed: " + sample_chat_summary
        mock_llm.invoke.return_value = mock_llm_response
        mock_get_llm.return_value = mock_llm
        
        mock_embeddings = Mock()
        test_embedding = [0.1] * EMBED_DIM
        mock_embeddings.embed_query.return_value = test_embedding
        mock_get_embeddings.return_value = mock_embeddings
        
        with patch.dict('os.environ', {'DATABASE_URL': test_db_url, 'EMBEDDING_DIMENSION': str(EMBED_DIM)}):
            mcp_server.init_database()
            result = mcp_server.store_chat_summary.fn(sample_chat_summary)
        
        # Verify result
        assert result["status"] == "success"
        assert "chat_id" in result
        assert "created_at" in result
        
        # Verify database
        conn = clean_test_db
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM chat_summaries WHERE id = %s;", (result["chat_id"],))
            db_result = cur.fetchone()
            assert db_result is not None
            assert "Processed:" in db_result['summary_text']
            assert db_result['metadata'] is not None
    
    @patch('mcp_server.get_llm')
    @patch('mcp_server.get_embeddings')
    def test_store_chat_summary_updates_digital_me(
        self, mock_get_embeddings, mock_get_llm,
        clean_test_db, test_db_url, sample_chat_summary
    ):
        """Test that storing a chat summary updates digital_me"""
        # Setup mocks
        mock_llm = Mock()
        
        # First call: process summary
        # Second call: update digital_me
        mock_llm_response1 = Mock()
        mock_llm_response1.content = "Processed summary"
        mock_llm_response2 = Mock()
        mock_llm_response2.content = "Updated digital me summary"
        mock_llm.invoke.side_effect = [mock_llm_response1, mock_llm_response2]
        mock_get_llm.return_value = mock_llm
        
        mock_embeddings = Mock()
        test_embedding = [0.1] * EMBED_DIM
        mock_embeddings.embed_query.return_value = test_embedding
        mock_get_embeddings.return_value = mock_embeddings
        
        with patch.dict('os.environ', {'DATABASE_URL': test_db_url, 'EMBEDDING_DIMENSION': str(EMBED_DIM)}):
            mcp_server.init_database()
            mcp_server.store_chat_summary.fn(sample_chat_summary)
        
        # Verify digital_me was updated
        conn = clean_test_db
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT summary_text FROM digital_me WHERE id = 1;")
            result = cur.fetchone()
            assert result is not None
            assert result['summary_text'] == "Updated digital me summary"


@pytest.mark.integration
class TestGetDigitalMeIntegration:
    """Integration tests for get_digital_me"""
    
    def test_get_digital_me_returns_empty_initially(self, clean_test_db, test_db_url):
        """Test that get_digital_me returns empty summary initially"""
        with patch.dict('os.environ', {'DATABASE_URL': test_db_url}):
            mcp_server.init_database()
            result = mcp_server.get_digital_me.fn()
        
        assert result["status"] == "success"
        assert result["summary"] == ""
    
    @patch('mcp_server.get_llm')
    @patch('mcp_server.get_embeddings')
    def test_get_digital_me_returns_updated_summary(
        self, mock_get_embeddings, mock_get_llm,
        clean_test_db, test_db_url, sample_chat_summary
    ):
        """Test that get_digital_me returns updated summary after storing"""
        # Setup mocks
        mock_llm = Mock()
        mock_llm_response1 = Mock()
        mock_llm_response1.content = "Processed summary"
        mock_llm_response2 = Mock()
        mock_llm_response2.content = "Updated digital me"
        mock_llm.invoke.side_effect = [mock_llm_response1, mock_llm_response2]
        mock_get_llm.return_value = mock_llm
        
        mock_embeddings = Mock()
        test_embedding = [0.1] * EMBED_DIM
        mock_embeddings.embed_query.return_value = test_embedding
        mock_get_embeddings.return_value = mock_embeddings
        
        with patch.dict('os.environ', {'DATABASE_URL': test_db_url, 'EMBEDDING_DIMENSION': str(EMBED_DIM)}):
            mcp_server.init_database()
            mcp_server.store_chat_summary.fn(sample_chat_summary)
            result = mcp_server.get_digital_me.fn()
        
        assert result["status"] == "success"
        assert result["summary"] == "Updated digital me"
        assert "updated_at" in result


@pytest.mark.integration
class TestAnswerQuestionIntegration:
    """Integration tests for answer_question"""
    
    @patch('mcp_server.get_llm')
    @patch('mcp_server.get_embeddings')
    def test_answer_question_with_stored_summaries(
        self, mock_get_embeddings, mock_get_llm,
        clean_test_db, test_db_url, sample_chat_summary
    ):
        """Test answering questions using stored summaries"""
        # Setup mocks for storing summaries
        mock_llm_store = Mock()
        mock_llm_response1 = Mock()
        mock_llm_response1.content = "Processed: " + sample_chat_summary
        mock_llm_response2 = Mock()
        mock_llm_response2.content = "Digital me summary"
        mock_llm_store.invoke.side_effect = [mock_llm_response1, mock_llm_response2]
        
        # Setup mock for answering question
        mock_llm_answer = Mock()
        mock_llm_answer_response = Mock()
        mock_llm_answer_response.content = "Yes, based on the stored summaries."
        mock_llm_answer.invoke.return_value = mock_llm_answer_response
        
        # Use different mocks for different calls
        call_tracker = {"count": 0}

        def get_llm_side_effect():
            call_tracker["count"] += 1
            if call_tracker["count"] <= 2:
                return mock_llm_store
            return mock_llm_answer
        
        mock_get_llm.side_effect = get_llm_side_effect
        
        mock_embeddings = Mock()
        test_embedding = [0.1] * EMBED_DIM
        mock_embeddings.embed_query.return_value = test_embedding
        mock_get_embeddings.return_value = mock_embeddings
        
        with patch.dict('os.environ', {'DATABASE_URL': test_db_url, 'EMBEDDING_DIMENSION': str(EMBED_DIM)}):
            mcp_server.init_database()
            # Store a summary first
            mcp_server.store_chat_summary.fn(sample_chat_summary)
            
            # Now answer a question
            result = mcp_server.answer_question.fn("Is Andras interested in quantum computing?")
        
        assert result["status"] == "success"
        assert "answer" in result
        assert result["context_sources"] > 0
    
    @patch('mcp_server.get_llm')
    @patch('mcp_server.get_embeddings')
    def test_answer_question_with_no_context(
        self, mock_get_embeddings, mock_get_llm,
        clean_test_db, test_db_url
    ):
        """Test answering questions when no summaries are stored"""
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = "I don't have enough information."
        mock_llm.invoke.return_value = mock_llm_response
        mock_get_llm.return_value = mock_llm
        
        mock_embeddings = Mock()
        test_embedding = [0.1] * EMBED_DIM
        mock_embeddings.embed_query.return_value = test_embedding
        mock_get_embeddings.return_value = mock_embeddings
        
        with patch.dict('os.environ', {'DATABASE_URL': test_db_url, 'EMBEDDING_DIMENSION': str(EMBED_DIM)}):
            mcp_server.init_database()
            result = mcp_server.answer_question.fn("Is Andras interested in quantum computing?")
        
        assert result["status"] == "success"
        assert "answer" in result


@pytest.mark.integration
class TestVectorSearch:
    """Test vector similarity search functionality"""
    
    @patch('mcp_server.get_llm')
    @patch('mcp_server.get_embeddings')
    def test_vector_similarity_search(
        self, mock_get_embeddings, mock_get_llm,
        clean_test_db, test_db_url
    ):
        """Test that vector similarity search works correctly"""
        # Create different embeddings for different summaries
        embedding1 = [0.9] * EMBED_DIM  # Similar to query
        embedding2 = [0.1] * EMBED_DIM  # Different from query
        
        mock_llm = Mock()
        mock_llm_response1 = Mock()
        mock_llm_response1.content = "Summary about quantum computing"
        mock_llm_response2 = Mock()
        mock_llm_response2.content = "Summary about cooking"
        mock_llm_response3 = Mock()
        mock_llm_response3.content = "Answer about quantum computing"
        mock_llm.invoke.side_effect = [mock_llm_response1, mock_llm_response2, mock_llm_response3]
        mock_get_llm.return_value = mock_llm
        
        mock_embeddings = Mock()
        # Return different embeddings for different queries
        def embed_side_effect(text):
            if "quantum" in text.lower():
                return embedding1
            return embedding2
        
        mock_embeddings.embed_query.side_effect = embed_side_effect
        mock_get_embeddings.return_value = mock_embeddings
        
        with patch.dict('os.environ', {'DATABASE_URL': test_db_url, 'EMBEDDING_DIMENSION': str(EMBED_DIM)}):
            mcp_server.init_database()
            
            # Store two summaries with different embeddings
            # We'll need to manually insert with specific embeddings for this test
            conn = clean_test_db
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_summaries (summary_text, embedding, metadata)
                    VALUES (%s, %s, %s), (%s, %s, %s);
                """, (
                    "Summary about quantum computing",
                    np.array(embedding1, dtype=np.float32),
                    '{}',
                    "Summary about cooking",
                    np.array(embedding2, dtype=np.float32),
                    '{}'
                ))
                conn.commit()
            
            # Answer a question about quantum computing
            result = mcp_server.answer_question.fn("Tell me about quantum computing")
        
        assert result["status"] == "success"
        # The answer should be based on the quantum computing summary
        assert "answer" in result

