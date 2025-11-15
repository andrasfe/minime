"""
Unit tests for recency weighting in retrieval.

Tests that recent conversations are prioritized appropriately.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRecencyWeighting:
    """Tests for recency-weighted retrieval."""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM client."""
        from langchain_core.messages import AIMessage
        llm = Mock()
        llm.invoke = Mock(return_value=AIMessage(content="Test answer"))
        return llm
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings client."""
        embeddings = Mock()
        embeddings.embed_query = Mock(return_value=[0.1] * 1024)
        return embeddings
    
    def test_recency_weight_zero_disables_boosting(self, mock_llm, mock_embeddings):
        """Test that recency_weight=0 uses pure semantic search."""
        from agents import run_subagent
        
        # Mock database with recent and old conversations
        mock_conn = Mock()
        mock_cursor = Mock()
        
        # Mock return: should use pure semantic order when recency_weight=0
        mock_cursor.fetchall = Mock(return_value=[
            {
                'summary_text': 'Old conversation (high semantic similarity)',
                'metadata': {},
                'base_similarity': 0.9,
                'boosted_similarity': 0.9,
                'days_old': None
            }
        ])
        mock_cursor.execute = Mock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_conn.cursor = Mock(return_value=mock_cursor)
        
        result = run_subagent(
            agent_id=0,
            variant="Test question",
            digital_me_summary="Test",
            llm_client=mock_llm,
            embeddings_client=mock_embeddings,
            db_connection=mock_conn,
            recency_weight=0.0,  # No recency weighting
            recency_decay_days=180
        )
        
        assert result.chunks_retrieved >= 0
        # Verify the query used pure semantic search (no exponential decay)
        execute_call = mock_cursor.execute.call_args
        query_sql = execute_call[0][0]
        assert "EXP" not in query_sql  # No exponential decay when recency_weight=0
    
    def test_recency_weight_positive_enables_boosting(self, mock_llm, mock_embeddings):
        """Test that recency_weight>0 uses recency-weighted search."""
        from agents import run_subagent
        
        mock_conn = Mock()
        mock_cursor = Mock()
        
        # Recent conversation should be boosted
        mock_cursor.fetchall = Mock(return_value=[
            {
                'summary_text': 'Recent conversation',
                'metadata': {'original_date': (datetime.now() - timedelta(days=30)).isoformat()},
                'base_similarity': 0.7,
                'boosted_similarity': 0.85,  # Boosted!
                'days_old': 30
            }
        ])
        mock_cursor.execute = Mock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_conn.cursor = Mock(return_value=mock_cursor)
        
        result = run_subagent(
            agent_id=0,
            variant="Test question",
            digital_me_summary="Test",
            llm_client=mock_llm,
            embeddings_client=mock_embeddings,
            db_connection=mock_conn,
            recency_weight=0.3,  # Enable recency weighting
            recency_decay_days=180
        )
        
        assert result.chunks_retrieved >= 0
        # Verify the query included recency boosting
        execute_call = mock_cursor.execute.call_args
        query_sql = execute_call[0][0]
        assert "boosted_similarity" in query_sql
        assert "EXP" in query_sql  # Exponential decay function
    
    def test_run_agentic_rag_with_recency(self, mock_llm, mock_embeddings):
        """Test full RAG pipeline with recency weighting."""
        from agents import run_agentic_rag
        from langchain_core.messages import AIMessage
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall = Mock(return_value=[
            {
                'summary_text': 'Test',
                'metadata': {},
                'base_similarity': 0.8,
                'boosted_similarity': 0.85,
                'days_old': 30
            }
        ])
        mock_cursor.fetchone = Mock(return_value={'summary_text': 'Digital twin summary'})
        mock_cursor.execute = Mock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_conn.cursor = Mock(return_value=mock_cursor)
        
        result = run_agentic_rag(
            question="Test question",
            llm_client=mock_llm,
            embeddings_client=mock_embeddings,
            db_connection=mock_conn,
            digital_me_summary="Test",
            depth=0,
            recency_weight=0.3,
            recency_decay_days=180
        )
        
        assert result["status"] == "success"
        assert result["recency_weight"] == 0.3
        assert result["recency_decay_days"] == 180
    
    def test_recency_metadata_omitted_when_disabled(self, mock_llm, mock_embeddings):
        """Test that recency metadata is omitted when recency_weight=0."""
        from agents import run_agentic_rag
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall = Mock(return_value=[])
        mock_cursor.fetchone = Mock(return_value={'summary_text': ''})
        mock_cursor.execute = Mock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_conn.cursor = Mock(return_value=mock_cursor)
        
        result = run_agentic_rag(
            question="Test question",
            llm_client=mock_llm,
            embeddings_client=mock_embeddings,
            db_connection=mock_conn,
            digital_me_summary="Test",
            depth=0,
            recency_weight=0.0,  # Disabled
            recency_decay_days=180
        )
        
        assert result["recency_weight"] is None
        assert result["recency_decay_days"] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

