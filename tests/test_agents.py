"""
Unit tests for parallel agentic RAG system.

Tests the parallel sub-agent execution.
"""

import pytest
from unittest.mock import Mock
from langchain_core.messages import AIMessage
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestParallelAgents:
    """Tests for parallel LangGraph agentic RAG."""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM client."""
        llm = Mock()
        response = AIMessage(content="Test answer about Andras's interests.")
        llm.invoke = Mock(return_value=response)
        return llm
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings client."""
        embeddings = Mock()
        embeddings.embed_query = Mock(return_value=[0.1] * 1024)
        return embeddings
    
    @pytest.fixture
    def mock_db_conn(self):
        """Mock database connection."""
        conn = Mock()
        cursor = Mock()
        
        # Mock chunks from vector search
        cursor.fetchall = Mock(return_value=[
            {
                'summary_text': 'User is interested in quantum computing.',
                'metadata': {},
                'similarity': 0.85
            },
            {
                'summary_text': 'User discussed AI projects.',
                'metadata': {},
                'similarity': 0.78
            }
        ])
        cursor.execute = Mock()
        cursor.__enter__ = Mock(return_value=cursor)
        cursor.__exit__ = Mock(return_value=False)
        
        conn.cursor = Mock(return_value=cursor)
        return conn
    
    def test_run_single_agent(self, mock_llm, mock_embeddings, mock_db_conn):
        """Test with single agent (depth=0)."""
        from agents import run_agentic_rag
        
        result = run_agentic_rag(
            question="What are my interests?",
            llm_client=mock_llm,
            embeddings_client=mock_embeddings,
            db_connection=mock_db_conn,
            digital_me_summary="Test summary",
            depth=0
        )
        
        assert result["status"] == "success"
        assert result["depth"] == 0
        assert result["num_agents"] == 1
        assert "answer" in result
    
    def test_run_parallel_agents(self, mock_llm, mock_embeddings, mock_db_conn):
        """Test with parallel agents (depth=3)."""
        from agents import run_agentic_rag
        from langchain_core.messages import AIMessage
        import json
        
        # Mock variant generation
        variant_response = AIMessage(content=json.dumps([
            "What interests does the user have?",
            "What hobbies does the user pursue?",
            "What topics fascinate the user?"
        ]))
        
        answer_response = AIMessage(content="Sub-agent answer")
        aggregation_response = AIMessage(content="Aggregated final answer")
        
        # Setup mock to return different responses
        mock_llm.invoke = Mock(side_effect=[variant_response, answer_response, answer_response, answer_response, aggregation_response])
        
        result = run_agentic_rag(
            question="What are my interests?",
            llm_client=mock_llm,
            embeddings_client=mock_embeddings,
            db_connection=mock_db_conn,
            digital_me_summary="Test summary",
            depth=3
        )
        
        assert result["status"] == "success"
        assert result["depth"] == 3
        assert result["num_agents"] == 3
        assert len(result["variants"]) == 3
        assert result["execution_mode"] == "langgraph_parallel_agents"
    
    def test_generate_query_variants(self, mock_llm):
        """Test query variant generation."""
        from agents import generate_query_variants
        from langchain_core.messages import AIMessage
        import json
        
        mock_llm.invoke = Mock(return_value=AIMessage(content=json.dumps([
            "Variant 1",
            "Variant 2",
            "Variant 3"
        ])))
        
        variants = generate_query_variants("Test question", 3, mock_llm)
        
        assert len(variants) == 3
        assert all(isinstance(v, str) for v in variants)
    
    def test_run_subagent(self, mock_llm, mock_embeddings, mock_db_conn):
        """Test individual sub-agent execution."""
        from agents import run_subagent
        
        result = run_subagent(
            agent_id=0,
            variant="What are my interests?",
            digital_me_summary="Test summary",
            llm_client=mock_llm,
            embeddings_client=mock_embeddings,
            db_connection=mock_db_conn
        )
        
        assert result.agent_id == 0
        assert result.variant_prompt == "What are my interests?"
        assert result.chunks_retrieved >= 0
        assert 0 <= result.confidence_score <= 1
        assert len(result.answer) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

