# Test Suite for Digital Me MCP Server

This directory contains comprehensive unit and integration tests for the MCP server.

## Test Structure

- **`conftest.py`**: Pytest fixtures and configuration
- **`test_unit.py`**: Unit tests using mocks (no database required)
- **`test_integration.py`**: Integration tests using a real test database
- **`test_docker_integration.py`**: Docker integration tests

## Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Make sure virtual environment is activated
pip install --upgrade pip
pip install -r requirements.txt
```

## Running Tests

### Run All Tests

```bash
# Make sure virtual environment is activated
pytest
```

### Run Only Unit Tests

```bash
pytest tests/test_unit.py
```

### Run Only Integration Tests

```bash
export TEST_DATABASE_URL=postgresql://user:password@localhost:5432/test_db
pytest tests/test_integration.py -m integration
```

### Run Docker Integration Tests

```bash
pytest tests/test_docker_integration.py -m docker -v
```

### Run with Coverage

```bash
pytest --cov=mcp_server --cov-report=html
```

Coverage report will be generated in `htmlcov/index.html`

### Complete Test Workflow Example

```bash
# 1. Create and activate venv
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run unit tests (fast, no external dependencies)
pytest tests/test_unit.py -v

# 4. Run integration tests (requires database)
export TEST_DATABASE_URL=postgresql://user:password@localhost:5432/test_db
pytest tests/test_integration.py -m integration -v

# 5. Run Docker tests (requires Docker)
pytest tests/test_docker_integration.py -m docker -v

# 6. Run all tests with coverage
pytest --cov=mcp_server --cov-report=term-missing --cov-report=html

# 7. Deactivate venv when done
deactivate
```

## Test Configuration

### Unit Tests

Unit tests use mocks and don't require a database. They test:
- LLM and embeddings initialization
- Tool functions with mocked dependencies
- Error handling
- Function logic

### Integration Tests

Integration tests require a test database with pgvector extension. Set the environment variable:

```bash
export TEST_DATABASE_URL=postgresql://user:password@localhost:5432/test_db
```

Integration tests verify:
- Database schema initialization
- End-to-end tool workflows
- Vector similarity search
- Data persistence
- Digital me updates

## Test Fixtures

### `mock_llm`
Mock LLM client for unit tests

### `mock_embeddings`
Mock embeddings client for unit tests

### `test_db_url`
Test database URL from environment (skips tests if not set)

### `test_db_connection`
Test database connection with pgvector registered

### `clean_test_db`
Test database connection with automatic cleanup before/after tests

### `sample_chat_summary`
Sample chat summary text for testing

### `sample_digital_me_summary`
Sample digital me summary for testing

### `reset_globals`
Automatically resets global variables before each test

## Writing New Tests

### Unit Test Example

```python
@patch('mcp_server.get_llm')
def test_my_function(mock_get_llm):
    mock_llm = Mock()
    mock_get_llm.return_value = mock_llm
    
    result = mcp_server.my_function()
    
    assert result["status"] == "success"
```

### Integration Test Example

```python
@pytest.mark.integration
def test_my_integration(clean_test_db, test_db_url):
    with patch.dict('os.environ', {'DATABASE_URL': test_db_url}):
        mcp_server.init_database()
        result = mcp_server.my_function()
    
    assert result["status"] == "success"
```

## Test Coverage Goals

- **Unit Tests**: 80%+ coverage of tool functions
- **Integration Tests**: 100% coverage of database operations
- **Error Handling**: All error paths tested

## Continuous Integration

Tests are designed to run in CI/CD pipelines:
- Unit tests run without external dependencies
- Integration tests can be skipped if `TEST_DATABASE_URL` is not set
- Coverage reports are generated automatically

