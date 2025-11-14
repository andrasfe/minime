# Digital Me MCP Server

Extract your "digital twin" from existing popular chatbots (ChatGPT, Claude) and create a searchable knowledge base about yourself.

## What This Is Good For

Have you had thousands of conversations with ChatGPT or Claude? This tool extracts all that knowledge into a single searchable database - your "digital twin". You can:

- **Import your chat history** from OpenAI (ChatGPT) and Anthropic (Claude)
- **Ask questions** about yourself: "What are my interests?", "What did I say about quantum computing?"
- **Get comprehensive answers** based on all your past conversations
- **Build a persistent knowledge base** that grows with every conversation

Perfect for creating a personal AI assistant that knows everything about you from your chat history.

## Quick Start: Database and MCP Server

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ with pgvector extension
- OpenAI API key OR Anthropic API key

### Step 1: Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Configure Environment

Create a `.env` file in the project root:

```env
# Database connection
DATABASE_URL=postgresql://digitalme:digitalme@localhost:5432/digitalme

# LLM Provider (choose one: 'openai' or 'anthropic')
LLM_PROVIDER=openai

# OpenAI Configuration (if using OpenAI)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small

# Anthropic Configuration (if using Anthropic)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Embeddings Provider
EMBEDDINGS_PROVIDER=openai
```

### Step 3: Start PostgreSQL

**Option A: Using Docker (Easiest)**
```bash
docker-compose up -d postgres
```

**Option B: Local PostgreSQL**
```bash
# Make sure PostgreSQL is running and create the database
createdb digitalme
psql -d digitalme -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Step 4: Start MCP Server

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Start the server (runs on http://127.0.0.1:8000/mcp)
python mcp_server.py
```

The server will automatically initialize the database schema and create necessary tables.

## Loading Your Chat History

Import your conversations from ChatGPT or Claude into your digital twin database.

### Prepare Your Exports

1. **OpenAI (ChatGPT)**: Export your conversations from ChatGPT settings
2. **Anthropic (Claude)**: Export your conversations from Claude settings

Place the exported files in the appropriate directory:

```
history/
├── openAI/          # Put ChatGPT exports here
│   └── [your exported files]
└── anthropic/       # Put Claude exports here
    └── [your exported files]
```

### Load History

**Load OpenAI conversations:**
```bash
python scripts/load_history.py --provider openai
```

**Load Anthropic conversations:**
```bash
python scripts/load_history.py --provider anthropic
```

**Preview before loading (dry run):**
```bash
python scripts/load_history.py --provider anthropic --dry-run
```

The script will:
- Process all conversations automatically
- Skip duplicates (already loaded conversations)
- Show progress and summary

## Running Inference (Asking Questions)

Ask questions about yourself from the command line:

```bash
python scripts/minime_client.py --question "What domains of quantum research am I focused on?"
```

**Output format options:**
```bash
# Human-readable (default)
python scripts/minime_client.py --question "What are my main interests?"

# JSON format
python scripts/minime_client.py --question "What are my main interests?" --format json

# Custom server URL
python scripts/minime_client.py --question "..." --url http://localhost:9000/mcp
```

The system searches through all your imported conversations and provides comprehensive answers based on your chat history.

## MCP Server Integration

The MCP server can be integrated into chatbots and AI assistants (Claude Desktop, ChatGPT, etc.) to provide your digital twin as context. Integration details and configuration examples will be documented separately.

**Basic integration:** Add the MCP server to your chatbot's configuration file. The server exposes three main tools:
- `store_chat_summary` - Store new conversations
- `get_digital_me` - Retrieve your digital twin profile
- `answer_question` - Ask questions about yourself

## Backup and Restore

### Create Backup

```bash
# Using the backup script (recommended)
./scripts/backup_db.sh

# Or specify custom location
./scripts/backup_db.sh backups/my_backup.sql
```

Backups are automatically compressed and saved to the `backups/` directory.

### Restore from Backup

```bash
# Using the restore script (with confirmation)
./scripts/restore_db.sh backups/backup_20241110_120000.sql.gz
```

**⚠️ WARNING:** Restoring will replace all existing data. Always backup first!

### Manual Backup/Restore

**Backup:**
```bash
mkdir -p backups
docker-compose exec postgres pg_dump -U digitalme -d digitalme --clean --if-exists > backups/backup_$(date +%Y%m%d_%H%M%S).sql
gzip backups/backup_*.sql
```

**Restore:**
```bash
gunzip -c backups/backup_20241110_120000.sql.gz | docker-compose exec -T postgres psql -U digitalme -d digitalme
```

---

## Docker Setup (Advanced)

For users who prefer Docker for everything.

### Quick Start with Docker

```bash
# 1. Create .env file (see Step 2 above)

# 2. Start all services
docker-compose up -d --build

# 3. Verify services are running
docker-compose ps

# 4. View logs
docker-compose logs -f mcp-server
docker-compose logs -f postgres

# 5. Stop services
docker-compose down

# 6. Stop and remove volumes (deletes data)
docker-compose down -v
```

### Docker Architecture

- **PostgreSQL with pgvector**: Database with vector extension (port 5432)
- **MCP Server**: Application server (port 8000)
- **Persistent Volumes**: Database data persists across restarts
- **Docker Network**: Isolated network for container communication

### Persistent Storage

Database files are stored in Docker volume `digital-me-postgres-data`:
- Data persists when containers are stopped/restarted
- Data persists when containers are recreated
- Data is only deleted when explicitly removing volumes

**View volumes:**
```bash
docker volume ls
docker volume inspect digital-me-postgres-data
```

### Docker Backup/Restore

**Backup:**
```bash
docker-compose exec postgres pg_dump -U digitalme -d digitalme --clean --if-exists > backups/backup_$(date +%Y%m%d_%H%M%S).sql
gzip backups/backup_*.sql
```

**Restore:**
```bash
gunzip -c backups/backup_20241110_120000.sql.gz | docker-compose exec -T postgres psql -U digitalme -d digitalme
```

### Docker Troubleshooting

**Check if services are running:**
```bash
docker-compose ps
docker-compose exec postgres pg_isready -U digitalme
```

**View logs:**
```bash
docker-compose logs postgres
docker-compose logs mcp-server
```

**Rebuild containers:**
```bash
docker-compose build --no-cache
docker-compose up -d
```

**Reset everything (WARNING: deletes data):**
```bash
docker-compose down -v
docker-compose up -d
```

---

## Technical Details

### Database Schema

**`chat_summaries` Table:**
- Stores individual chat summaries with vector embeddings
- Fields: `id`, `summary_text`, `embedding`, `metadata`, `created_at`, `updated_at`

**`digital_me` Table:**
- Single record containing comprehensive user profile
- Fields: `id` (always 1), `summary_text`, `embedding`, `updated_at`

### MCP Tools

1. **`store_chat_summary(summary: str)`** - Store a chat summary in the database
2. **`get_digital_me()`** - Retrieve the comprehensive digital me summary
3. **`answer_question(question: str)`** - Answer questions using RAG retrieval

### Architecture

This server follows ZFC (Zero Framework Cognition) principles - pure orchestration that delegates ALL reasoning to external AI. The server provides:
- Database persistence (PostgreSQL/pgvector)
- LLM integration (OpenAI or Anthropic via LangChain)
- Vector embeddings for semantic search
- MCP protocol interface (streamable HTTP)

All reasoning, processing, and decision-making is handled by external AI models.

## Testing

```bash
# Run all tests
pytest

# Run only unit tests (no database required)
pytest tests/test_unit.py

# Run only integration tests (requires test database)
export TEST_DATABASE_URL=postgresql://user:password@localhost:5432/test_db
pytest tests/test_integration.py -m integration

# Run with coverage
pytest --cov=mcp_server --cov-report=html
```

See `tests/README.md` for detailed testing documentation.

## License

MIT
