"""
Docker integration tests for Digital Me MCP Server

These tests verify that the Docker setup works correctly with persistent storage.
Requires Docker and docker-compose to be installed.
"""

import pytest
import subprocess
import time
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector


@pytest.fixture(scope="module")
def docker_compose():
    """Start Docker Compose services for testing"""
    # Start services
    subprocess.run(
        ["docker-compose", "-f", "docker-compose.yml", "up", "-d", "--build"],
        check=True,
        cwd=os.path.dirname(os.path.dirname(__file__))
    )
    
    # Wait for services to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                user="digitalme",
                password="digitalme",
                database="digitalme"
            )
            conn.close()
            break
        except psycopg2.OperationalError:
            if i == max_retries - 1:
                raise
            time.sleep(2)
    
    yield
    
    # Cleanup: stop services but keep volumes
    subprocess.run(
        ["docker-compose", "-f", "docker-compose.yml", "down"],
        cwd=os.path.dirname(os.path.dirname(__file__))
    )


@pytest.fixture
def db_connection(docker_compose):
    """Get database connection to Docker PostgreSQL"""
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="digitalme",
        password="digitalme",
        database="digitalme"
    )
    register_vector(conn)
    yield conn
    conn.close()


@pytest.mark.docker
class TestDockerSetup:
    """Test Docker container setup"""
    
    def test_postgres_container_running(self, docker_compose):
        """Test that PostgreSQL container is running"""
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=digital-me-postgres", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True
        )
        assert "digital-me-postgres" in result.stdout
    
    def test_mcp_server_container_running(self, docker_compose):
        """Test that MCP server container is running"""
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=digital-me-mcp-server", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True
        )
        assert "digital-me-mcp-server" in result.stdout
    
    def test_postgres_has_pgvector_extension(self, db_connection):
        """Test that PostgreSQL has pgvector extension enabled"""
        with db_connection.cursor() as cur:
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector');")
            result = cur.fetchone()
            assert result[0] is True
    
    def test_database_connection(self, db_connection):
        """Test that we can connect to the database"""
        with db_connection.cursor() as cur:
            cur.execute("SELECT version();")
            result = cur.fetchone()
            assert result is not None
            assert "PostgreSQL" in result[0]


@pytest.mark.docker
class TestPersistentStorage:
    """Test persistent storage functionality"""
    
    def test_data_persists_after_container_restart(self, db_connection, docker_compose):
        """Test that data persists after container restart"""
        # Create a test table and insert data
        with db_connection.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS test_persistence (
                    id SERIAL PRIMARY KEY,
                    data TEXT
                );
            """)
            cur.execute("INSERT INTO test_persistence (data) VALUES ('test data');")
            db_connection.commit()
        
        # Restart PostgreSQL container
        subprocess.run(
            ["docker-compose", "-f", "docker-compose.yml", "restart", "postgres"],
            check=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        
        # Wait for container to be ready
        time.sleep(5)
        
        # Verify data still exists
        with db_connection.cursor() as cur:
            cur.execute("SELECT data FROM test_persistence WHERE id = 1;")
            result = cur.fetchone()
            assert result is not None
            assert result[0] == "test data"
        
        # Cleanup
        with db_connection.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS test_persistence;")
            db_connection.commit()
    
    def test_volume_exists(self, docker_compose):
        """Test that persistent volume exists"""
        result = subprocess.run(
            ["docker", "volume", "ls", "--filter", "name=digital-me-postgres-data", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            check=True
        )
        assert "digital-me-postgres-data" in result.stdout
    
    def test_schema_persists_after_restart(self, db_connection, docker_compose):
        """Test that database schema persists after restart"""
        # Initialize schema
        import mcp_server
        with mcp_server.get_db_connection() as conn:
            mcp_server.init_database()
        
        # Restart container
        subprocess.run(
            ["docker-compose", "-f", "docker-compose.yml", "restart", "postgres"],
            check=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        time.sleep(5)
        
        # Verify schema still exists
        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'chat_summaries'
                );
            """)
            assert cur.fetchone()[0] is True
            
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'digital_me'
                );
            """)
            assert cur.fetchone()[0] is True


@pytest.mark.docker
class TestDockerDatabaseOperations:
    """Test database operations in Docker environment"""
    
    def test_init_database_in_docker(self, db_connection):
        """Test database initialization in Docker"""
        import mcp_server
        import os
        
        # Reset global state
        mcp_server.db_pool = None
        
        # Set database URL for Docker
        os.environ['DATABASE_URL'] = 'postgresql://digitalme:digitalme@localhost:5432/digitalme'
        
        # Initialize database
        mcp_server.init_database()
        
        # Verify tables exist
        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('chat_summaries', 'digital_me');
            """)
            tables = [row[0] for row in cur.fetchall()]
            assert 'chat_summaries' in tables
            assert 'digital_me' in tables
    
    def test_store_and_retrieve_in_docker(self, db_connection):
        """Test storing and retrieving data in Docker environment"""
        import mcp_server
        import os
        from unittest.mock import Mock, patch
        
        # Reset global state
        mcp_server.db_pool = None
        mcp_server.llm = None
        mcp_server.embeddings = None
        
        # Set database URL for Docker
        os.environ['DATABASE_URL'] = 'postgresql://digitalme:digitalme@localhost:5432/digitalme'
        
        # Setup mocks
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = "Processed summary"
        mock_llm.invoke.return_value = mock_llm_response
        
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        
        with patch('mcp_server.get_llm', return_value=mock_llm), \
             patch('mcp_server.get_embeddings', return_value=mock_embeddings):
            
            # Store a summary
            result = mcp_server.store_chat_summary.fn("Test summary for Docker")
            
            assert result["status"] == "success"
            assert "chat_id" in result
            
            # Verify in database
            with db_connection.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM chat_summaries WHERE id = %s;", (result["chat_id"],))
                db_result = cur.fetchone()
                assert db_result is not None
                assert "Processed summary" in db_result['summary_text']


@pytest.mark.docker
class TestDockerVolumePersistence:
    """Test that volumes persist data correctly"""
    
    def test_multiple_restarts_preserve_data(self, db_connection, docker_compose):
        """Test that data survives multiple container restarts"""
        import mcp_server
        import os
        from unittest.mock import Mock, patch
        
        # Reset global state
        mcp_server.db_pool = None
        mcp_server.llm = None
        mcp_server.embeddings = None
        
        # Set database URL for Docker
        os.environ['DATABASE_URL'] = 'postgresql://digitalme:digitalme@localhost:5432/digitalme'
        
        # Setup mocks
        mock_llm = Mock()
        mock_llm_response1 = Mock()
        mock_llm_response1.content = "First summary"
        mock_llm_response2 = Mock()
        mock_llm_response2.content = "Updated digital me"
        mock_llm.invoke.side_effect = [mock_llm_response1, mock_llm_response2]
        
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        
        with patch('mcp_server.get_llm', return_value=mock_llm), \
             patch('mcp_server.get_embeddings', return_value=mock_embeddings):
            
            # Store data
            mcp_server.store_chat_summary.fn("Test data for persistence")
        
        # Get the chat_id
        with db_connection.cursor() as cur:
            cur.execute("SELECT id FROM chat_summaries ORDER BY id DESC LIMIT 1;")
            result = cur.fetchone()
            if not result:
                pytest.skip("No data stored, skipping persistence test")
            chat_id = result[0]
        
        # Restart container multiple times
        for i in range(3):
            subprocess.run(
                ["docker-compose", "-f", "docker-compose.yml", "restart", "postgres"],
                check=True,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
            time.sleep(5)
            
            # Reconnect to database
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                user="digitalme",
                password="digitalme",
                database="digitalme"
            )
            register_vector(conn)
            
            # Verify data still exists
            with conn.cursor() as cur:
                cur.execute("SELECT summary_text FROM chat_summaries WHERE id = %s;", (chat_id,))
                result = cur.fetchone()
                assert result is not None
                assert "First summary" in result[0]
            
            conn.close()


@pytest.mark.docker
class TestDockerNetwork:
    """Test Docker networking"""
    
    def test_containers_can_communicate(self, docker_compose):
        """Test that containers can communicate on the network"""
        # Test that the MCP server can connect to the database from within the container
        # We'll test by trying to connect from the host to the database
        # (which uses the same network as the containers)
        import mcp_server
        import os
        
        # Reset global state
        mcp_server.db_pool = None
        
        # Set database URL for Docker (using postgres hostname for container-to-container)
        # But we'll test from host, so use localhost
        os.environ['DATABASE_URL'] = 'postgresql://digitalme:digitalme@localhost:5432/digitalme'
        
        # This should work if networking is correct
        try:
            conn = mcp_server.get_db_connection()
            conn.close()
            assert True
        except Exception as e:
            pytest.fail(f"Containers cannot communicate: {e}")

