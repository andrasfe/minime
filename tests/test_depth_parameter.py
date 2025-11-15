"""
Unit tests for depth parameter functionality.

Tests the depth parameter in answer_question tool and multi-agent system.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDepthParameter:
    """Tests for depth parameter in answer_question tool."""
    
    @pytest.fixture
    def mock_db_initialized(self):
        """Mock database initialization."""
        with patch('mcp_server.init_database'):
            yield
    
    @pytest.fixture
    def mock_single_agent(self):
        """Mock single-agent answer function."""
        with patch('mcp_server._answer_question_single_agent') as mock:
            mock.return_value = {
                "status": "success",
                "answer": "Single agent answer",
                "depth": 0,
                "execution_mode": "single_agent"
            }
            yield mock
    
    @pytest.fixture
    def mock_multi_agent(self):
        """Mock multi-agent answer function."""
        with patch('mcp_server._answer_question_multi_agent') as mock:
            mock.return_value = {
                "status": "success",
                "answer": "Multi agent answer",
                "depth": 3,
                "execution_mode": "parallel"
            }
            yield mock
    
    def test_depth_0_uses_single_agent(self, mock_db_initialized, mock_single_agent, mock_multi_agent):
        """Test that depth=0 uses single-agent mode."""
        import mcp_server
        
        # Call the underlying function through the fn attribute
        result = mcp_server.answer_question.fn("Test question", depth=0)
        
        assert result["status"] == "success"
        assert result["depth"] == 0
        assert result["execution_mode"] == "single_agent"
        mock_single_agent.assert_called_once_with("Test question")
        mock_multi_agent.assert_not_called()
    
    def test_depth_1_uses_multi_agent(self, mock_db_initialized, mock_single_agent, mock_multi_agent):
        """Test that depth=1 uses multi-agent mode with 1 variant."""
        import mcp_server
        
        result = mcp_server.answer_question.fn("Test question", depth=1)
        
        assert result["status"] == "success"
        mock_multi_agent.assert_called_once_with("Test question", num_variants=1)
        mock_single_agent.assert_not_called()
    
    def test_depth_3_uses_multi_agent(self, mock_db_initialized, mock_single_agent, mock_multi_agent):
        """Test that depth=3 uses multi-agent mode with 3 variants."""
        import mcp_server
        
        result = mcp_server.answer_question.fn("Test question", depth=3)
        
        mock_multi_agent.assert_called_once_with("Test question", num_variants=3)
        mock_single_agent.assert_not_called()
    
    def test_depth_5_uses_multi_agent(self, mock_db_initialized, mock_single_agent, mock_multi_agent):
        """Test that depth=5 uses multi-agent mode with 5 variants."""
        import mcp_server
        
        result = mcp_server.answer_question.fn("Test question", depth=5)
        
        mock_multi_agent.assert_called_once_with("Test question", num_variants=5)
        mock_single_agent.assert_not_called()
    
    def test_depth_negative_clamped_to_0(self, mock_db_initialized, mock_single_agent, mock_multi_agent):
        """Test that negative depth is clamped to 0."""
        import mcp_server
        
        result = mcp_server.answer_question.fn("Test question", depth=-5)
        
        mock_single_agent.assert_called_once_with("Test question")
        mock_multi_agent.assert_not_called()
    
    def test_depth_too_high_clamped_to_5(self, mock_db_initialized, mock_single_agent, mock_multi_agent):
        """Test that depth > 5 is clamped to 5."""
        import mcp_server
        
        result = mcp_server.answer_question.fn("Test question", depth=10)
        
        mock_multi_agent.assert_called_once_with("Test question", num_variants=5)
        mock_single_agent.assert_not_called()
    
    def test_default_depth_is_0(self, mock_db_initialized, mock_single_agent, mock_multi_agent):
        """Test that default depth is 0 (single-agent)."""
        import mcp_server
        
        result = mcp_server.answer_question.fn("Test question")
        
        mock_single_agent.assert_called_once_with("Test question")
        mock_multi_agent.assert_not_called()


# Legacy agent tests removed - see test_langgraph_parallel.py for current tests


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

