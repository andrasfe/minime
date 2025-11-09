# Digital Me MCP Server

An intelligent MCP (Model Context Protocol) server that stores chat summaries in PostgreSQL/pgvector for RAG (Retrieval Augmented Generation) and maintains a "digital me" profile that continuously learns about you.

## Features

- **Store Chat Summaries**: Process and store chat summaries with LLM-optimized RAG formatting
- **Digital Me Profile**: Maintains a single comprehensive summary about you that updates with each new chat
- **Question Answering**: Answer ad-hoc questions about yourself using semantic search across all stored summaries

## Architecture

This server follows ZFC (Zero Framework Cognition) principles - it's pure orchestration that delegates ALL reasoning to external AI. The server provides:

- Database persistence (PostgreSQL/pgvector)
- LLM integration (OpenAI or Anthropic via LangChain)
- Vector embeddings for semantic search
- MCP protocol interface (streamable HTTP)

All reasoning, processing, and decision-making is handled by external AI models.

## Prerequisites

- Python 3.10+
- PostgreSQL 14+ with pgvector extension
- OpenAI API key OR Anthropic API key

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up PostgreSQL with pgvector

```bash
# Install pgvector extension in your PostgreSQL database
psql -d your_database -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/database_name

# LLM Provider (choose one: 'openai' or 'anthropic')
LLM_PROVIDER=openai

# OpenAI Configuration (if using OpenAI)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small

# Anthropic Configuration (if using Anthropic)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Embeddings Provider (currently only 'openai' supported)
EMBEDDINGS_PROVIDER=openai
```

### 4. Run the Server

The server can be run in two modes:

**Standard stdio mode (for MCP clients):**
```bash
python mcp_server.py
```

**HTTP mode (for streamable HTTP protocol):**
FastMCP automatically supports streamable HTTP when configured in your MCP client. The server will automatically:
- Initialize the database schema
- Create necessary tables and indexes
- Set up the pgvector extension

## MCP Tools

### 1. `store_chat_summary`

Stores a chat summary in the database for RAG retrieval.

**Input:**
- `summary` (string): A summary of a chat conversation

**Process:**
1. Uses LLM to process the summary into optimal RAG format
2. Generates embeddings for vector search
3. Stores in PostgreSQL/pgvector
4. Updates the "digital me" record with new information

**Output:**
- Status and details about the stored summary

### 2. `get_digital_me`

Retrieves the comprehensive "digital me" summary record.

**Output:**
- The current digital me summary and last update timestamp

### 3. `answer_question`

Answers ad-hoc questions about you using RAG retrieval.

**Input:**
- `question` (string): A question about you (e.g., "Is Andras interested in quantum computing?")

**Process:**
1. Searches chat summaries and digital_me using semantic search
2. Retrieves relevant context
3. Uses LLM to answer based on retrieved context

**Output:**
- Answer to the question and number of context sources used

## Database Schema

### `chat_summaries` Table

Stores individual chat summaries with vector embeddings:

- `id`: Primary key
- `summary_text`: LLM-processed summary text (optimized for RAG)
- `embedding`: Vector embedding (1536 dimensions for OpenAI)
- `metadata`: JSONB field for additional metadata
- `created_at`: Timestamp
- `updated_at`: Timestamp

### `digital_me` Table

Single record containing the comprehensive user profile:

- `id`: Always 1 (enforced by constraint)
- `summary_text`: Comprehensive summary about the user
- `embedding`: Vector embedding for semantic search
- `updated_at`: Last update timestamp

## Usage Example

### Connecting to MCP Client

Add to your MCP client configuration (e.g., `~/.config/claude/config.json`):

```json
{
  "mcpServers": {
    "digital-me": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "DATABASE_URL": "postgresql://user:password@localhost:5432/dbname",
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "your_key_here"
      }
    }
  }
}
```

### Using the Tools

Once connected, you can use the tools from your MCP client:

1. **Store a chat summary:**
   ```
   store_chat_summary("Had a conversation about quantum computing and machine learning. Discussed interest in building AI agents.")
   ```

2. **Get your digital profile:**
   ```
   get_digital_me()
   ```

3. **Ask a question:**
   ```
   answer_question("Is Andras interested in quantum computing?")
   ```

## ZFC Compliance

This server is designed according to ZFC (Zero Framework Cognition) principles:

✅ **Allowed (ZFC-Compliant):**
- IO and plumbing (database operations, file I/O)
- Structural safety checks (schema validation, required fields)
- Policy enforcement (timeouts, error handling)
- Mechanical transforms (parameter substitution, formatting)
- State management (database persistence)

❌ **Forbidden (ZFC-Violations):**
- Local intelligence/reasoning (all delegated to LLM)
- Ranking/scoring/selection algorithms
- Plan/composition/scheduling decisions
- Semantic analysis (handled by LLM)
- Heuristic classification

All reasoning is delegated to external AI models via LangChain.

## Testing

The project includes comprehensive unit and integration tests.

### Running Tests

**Run all tests:**
```bash
pytest
```

**Run only unit tests (no database required):**
```bash
pytest tests/test_unit.py
```

**Run only integration tests (requires test database):**
```bash
export TEST_DATABASE_URL=postgresql://user:password@localhost:5432/test_db
pytest tests/test_integration.py -m integration
```

**Run with coverage:**
```bash
pytest --cov=mcp_server --cov-report=html
```

See `tests/README.md` for detailed testing documentation.

## Docker Setup

The project includes Docker support with persistent storage for the database.

### Prerequisites

- Docker and Docker Compose installed
- Environment variables configured (see `.env.example`)

### Quick Start with Docker

1. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Start services:**
   ```bash
   docker-compose up -d --build
   ```

3. **Verify services are running:**
   ```bash
   docker-compose ps
   ```

4. **View logs:**
   ```bash
   docker-compose logs -f mcp-server
   docker-compose logs -f postgres
   ```

5. **Stop services:**
   ```bash
   docker-compose down
   ```

6. **Stop and remove volumes (deletes data):**
   ```bash
   docker-compose down -v
   ```

### Docker Architecture

The Docker setup includes:

- **PostgreSQL with pgvector**: Database with vector extension
- **MCP Server**: Application server
- **Persistent Volumes**: Database data persists across container restarts
- **Docker Network**: Isolated network for container communication

### Persistent Storage

Database files are stored in a Docker volume named `digital-me-postgres-data`. This ensures:

- Data persists when containers are stopped/restarted
- Data persists when containers are recreated
- Data is only deleted when explicitly removing volumes

**View volumes:**
```bash
docker volume ls
docker volume inspect digital-me-postgres-data
```

**Backup volume:**
```bash
docker run --rm -v digital-me-postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/backup.tar.gz /data
```

**Restore volume:**
```bash
docker run --rm -v digital-me-postgres-data:/data -v $(pwd):/backup alpine sh -c "cd /data && tar xzf /backup/backup.tar.gz"
```

### Docker Integration Tests

Run Docker integration tests to verify the setup:

```bash
# Run all Docker tests
pytest tests/test_docker_integration.py -m docker -v

# Or use the test script
./scripts/test_docker.sh
```

Docker tests verify:
- Containers start correctly
- PostgreSQL has pgvector extension
- Database operations work
- Data persists across restarts
- Network communication works

### Docker Compose Configuration

The `docker-compose.yml` file configures:

- **PostgreSQL service**: Port 5432, persistent volume
- **MCP Server service**: Depends on PostgreSQL, uses environment variables
- **Network**: Isolated bridge network
- **Volumes**: Named volume for database persistence

### Environment Variables in Docker

Environment variables can be set in:

1. `.env` file (loaded automatically by docker-compose)
2. `docker-compose.yml` (for defaults)
3. Command line: `docker-compose run -e VAR=value mcp-server`

### Troubleshooting

**Database connection issues:**
```bash
# Check if PostgreSQL is ready
docker-compose exec postgres pg_isready -U digitalme

# Check logs
docker-compose logs postgres
```

**Volume issues:**
```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect digital-me-postgres-data

# Remove and recreate volume (WARNING: deletes data)
docker-compose down -v
docker-compose up -d
```

**Rebuild containers:**
```bash
docker-compose build --no-cache
docker-compose up -d
```

## License

MIT

