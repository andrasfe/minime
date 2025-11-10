"""
Pytest configuration and fixtures for MCP server tests
"""

import os
import pytest
import psycopg2
from psycopg2.extras import RealDictCursor
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

# Load test environment variables
load_dotenv()

TEST_EMBEDDING_DIMENSION = 4096


@pytest.fixture(autouse=True)
def default_env(monkeypatch):
    """Ensure consistent environment configuration for tests."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "cohere-v2")
    monkeypatch.setenv("EMBEDDING_MODEL", "embed-english-v2.0")
    monkeypatch.setenv("EMBEDDING_DIMENSION", str(TEST_EMBEDDING_DIMENSION))
    yield


@pytest.fixture
def mock_llm():
    """Mock LLM client"""
    mock = Mock()
    mock_response = Mock()
    mock_response.content = "Processed summary text"
    mock.invoke.return_value = mock_response
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock embeddings client"""
    mock = Mock()
    # Return a mock embedding vector matching configured dimension
    mock.embed_query.return_value = [0.1] * TEST_EMBEDDING_DIMENSION
    return mock


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing"""
    return np.array([0.1] * TEST_EMBEDDING_DIMENSION, dtype=np.float32)


@pytest.fixture
def test_db_url():
    """Get test database URL from environment"""
    db_url = os.getenv("TEST_DATABASE_URL")
    if not db_url:
        pytest.skip("TEST_DATABASE_URL not set. Skipping integration tests.")
    return db_url


@pytest.fixture
def test_db_connection(test_db_url):
    """Create a test database connection"""
    conn = psycopg2.connect(test_db_url)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    register_vector(conn)
    yield conn
    conn.close()


@pytest.fixture
def clean_test_db(test_db_connection):
    """Clean test database before and after tests"""
    conn = test_db_connection
    with conn.cursor() as cur:
        # Drop tables if they exist
        cur.execute("DROP TABLE IF EXISTS chat_summaries CASCADE;")
        cur.execute("DROP TABLE IF EXISTS digital_me CASCADE;")
        conn.commit()
    
    yield conn
    
    # Clean up after test
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS chat_summaries CASCADE;")
        cur.execute("DROP TABLE IF EXISTS digital_me CASCADE;")
        conn.commit()


@pytest.fixture
def sample_chat_summary():
    """Sample chat summary for testing"""
    return "Had a conversation about quantum computing and machine learning. Discussed interest in building AI agents and working with MCP servers."


@pytest.fixture
def sample_digital_me_summary():
    """Sample digital me summary for testing"""
    return "Andras is interested in quantum computing, machine learning, and AI agents. He works with MCP servers and enjoys building intelligent systems."


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global variables before each test"""
    import mcp_server
    mcp_server.db_pool = None
    mcp_server.llm = None
    mcp_server.embeddings = None
    yield
    # Cleanup after test
    if mcp_server.db_pool:
        mcp_server.db_pool.closeall()
    mcp_server.db_pool = None
    mcp_server.llm = None
    mcp_server.embeddings = None

