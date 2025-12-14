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
- OpenRouter API key (for LLM)
- Cohere API key (for embeddings)

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

# OpenRouter Configuration (for LLM)
# OpenRouter allows you to use OpenAI, Anthropic, and other models
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-4o  # or anthropic/claude-3-5-sonnet, etc.
LLM_TEMPERATURE=0.7

# Cohere Configuration (for embeddings)
COHERE_API_KEY=your_cohere_api_key_here
EMBEDDING_PROVIDER=cohere-v2
EMBEDDING_MODEL=embed-english-v2.0
COHERE_TRUNCATE=END

# LangGraph Parallel Multi-Agent RAG Configuration (Optional)
# Models for parallel sub-agent execution
OPENROUTER_CENTRAL_AGENT_MODEL=openai/gpt-4o  # Model for all agents (generation & aggregation)
CENTRAL_AGENT_TEMPERATURE=0.7

# Recency Weighting Configuration (Optional)
# Prioritize recent conversations in retrieval
RECENCY_WEIGHT=0.3  # 0.0 = pure semantic, 1.0 = heavy recency bias (default: 0.3)
RECENCY_DECAY_DAYS=180  # Half-life for time decay in days (default: 180)
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

> **⚠️ WARNING**: Load process may be slow and costly depending on the model you choose for it!

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

### Incremental Updates

The system uses **content-based duplicate detection** (SHA-256 hash of each conversation). This means:

- **Run the script as many times as you want** — already-loaded conversations are automatically skipped
- **Overwrite your export files** — just export fresh from ChatGPT/Claude and replace the old files
- **No manual tracking needed** — the database knows what's already loaded

**Typical workflow for updates:**

1. Export your latest conversations from ChatGPT or Claude
2. Overwrite/replace the files in `history/openAI/` or `history/anthropic/`
3. Run the load script again

```bash
python scripts/load_history.py --provider openai
# Output: "50 loaded, 450 skipped, 0 failed"
```

Only the new conversations will be processed and loaded.

## Running Inference (Asking Questions)

Ask questions about yourself from the command line:

```bash
python scripts/minime_client.py --question "What domains of quantum research am I focused on?"
```

### Research Depth Parameter

Control how comprehensive your answers are with the `--depth` parameter:

```bash
# Basic mode (default): Single RAG query, fast and cost-effective
python scripts/minime_client.py --question "What are my interests?" --depth 0

# Deep mode: Multi-agent RAG with 3 parallel queries exploring different angles
python scripts/minime_client.py --question "What are my interests?" --depth 3

# Maximum depth: 5 parallel agents for most comprehensive answer
python scripts/minime_client.py --question "What are my interests?" --depth 5
```

**Depth Options:**
- `--depth 0` (default): Basic single-agent mode - fast, cost-effective
- `--depth 1-5`: Multi-agent mode with N parallel sub-agents
  - Higher depth = more comprehensive answers but higher cost and latency
  - Recommended: `--depth 3` for complex questions

### Retrieval Breadth Parameter (Map-Reduce)

For questions that require scanning many conversations (like "What are ALL my questions about X?"), use the `--breadth` parameter:

```bash
# Default: Scan up to 100 entries (no map-reduce)
python scripts/minime_client.py --question "What are my interests?"

# Use --breadth flag without value: defaults to 5 (500 entries)
python scripts/minime_client.py --question "What questions about south america did I have?" --breadth

# Scan 500 entries using map-reduce
python scripts/minime_client.py --question "What questions about south america did I have?" --breadth 5

# Maximum: Scan 1000 entries
python scripts/minime_client.py --question "List all my coding projects" --breadth 10
```

**Breadth Options:**
- No `--breadth` flag: Standard retrieval (up to 100 entries)
- `--breadth` (no value): Map-reduce with 500 entries (breadth=5)
- `--breadth 2-10`: Map-reduce mode scanning `breadth × 100` entries
  - Chunks entries into batches of 50
  - Processes batches in parallel
  - Aggregates results into final answer
  - Higher breadth = more comprehensive coverage but higher cost

**When to Use Breadth:**
- Questions asking for "all" or "every" instance of something
- Retrospective queries spanning many conversations
- When standard retrieval returns incomplete results due to the 100-entry limit

**Note:** `--breadth` and `--depth` serve different purposes:
- `--depth` controls query variant exploration (different angles on the same question)
- `--breadth` controls retrieval volume (how much of the database to scan)

**Other Output Options:**
```bash
# JSON format
python scripts/minime_client.py --question "What are my main interests?" --format json

# Custom server URL
python scripts/minime_client.py --question "..." --url http://localhost:9000/mcp
```

The system searches through all your imported conversations and provides comprehensive answers based on your chat history.

### Multi-Agent RAG with LangGraph (Advanced)

The system uses **LangGraph with parallel sub-agents** (based on [LangChain's agentic RAG pattern](https://docs.langchain.com/oss/python/langgraph/agentic-rag)) for comprehensive query processing:

**How it works:**

1. **Query Variant Generation**: Creates N different phrasings of your question
2. **Parallel Sub-Agent Execution**: Each sub-agent independently:
   - Retrieves top 100 relevant chunks from vector store
   - Generates an answer using retrieved context
   - Returns answer with confidence score
3. **Aggregation**: Central agent synthesizes all sub-agent answers into comprehensive final response

**Depth Parameter (Number of Parallel Sub-Agents):**

- `depth=0` or `depth=1`: Single agent (fast, cost-effective)
- `depth=2-5`: Multiple parallel sub-agents exploring different query angles
  - `depth=3` (recommended): Good balance of comprehensiveness and cost
  - `depth=5` (maximum): Most comprehensive but highest cost

**Benefits:**

- ✅ **Parallel execution**: Sub-agents run simultaneously for speed
- ✅ **Multiple perspectives**: Different query variants explore different aspects
- ✅ **Comprehensive answers**: Aggregation combines insights from all agents
- ✅ **Efficiency**: Each agent retrieves top 100 docs (not N×100 redundant retrievals)
- ✅ **Recency weighting**: Recent conversations are prioritized (configurable)

**Configuration:**

```env
# Models for parallel agents
OPENROUTER_CENTRAL_AGENT_MODEL=openai/gpt-4o  # For variant generation & aggregation
OPENROUTER_SUBAGENT_MODEL=openai/gpt-4o-mini  # Not used (same model for all)
```

**Recency Weighting:**

The system automatically prioritizes recent conversations using an exponential decay function:

- **`RECENCY_WEIGHT`** (0.0-1.0): Controls how much to boost recent conversations
  - `0.0`: Pure semantic search (no recency bias)
  - `0.3` (default): Balanced - recent conversations get moderate boost
  - `1.0`: Heavy recency bias - strongly favor recent conversations

- **`RECENCY_DECAY_DAYS`** (default: 180): Half-life for time decay
  - Conversations older than this get exponentially less boost
  - 180 days = 6 months

**Formula:** `boosted_similarity = base_similarity × (1 + recency_weight × exp(-days_old / recency_decay_days))`

**Example:**
- A conversation from 30 days ago with 0.7 semantic similarity → ~0.88 boosted similarity (with default settings)
- A conversation from 1 year ago with 0.7 semantic similarity → ~0.75 boosted similarity

**Example Query:**

```bash
# Single agent
python scripts/minime_client.py --question "What are my interests?" --depth 0

# 3 parallel agents (recommended)
python scripts/minime_client.py --question "What are my interests?" --depth 3
```

**Output includes metadata:**

```json
{
  "answer": "Comprehensive synthesized answer...",
  "execution_mode": "langgraph_parallel_agents",
  "num_agents": 3,
  "depth": 3,
  "total_chunks": 300,
  "avg_confidence": 0.82,
  "variants": ["variant 1", "variant 2", "variant 3"],
  "recency_weight": 0.3,
  "recency_decay_days": 180
}
```

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

### Clear Database

To delete all data and start fresh:

```bash
# Using the clear script (with confirmation prompt)
./scripts/clear_db.sh
```

**⚠️ WARNING:** This will permanently delete:
- All chat summaries
- Digital me profile (reset to empty)

The database schema will remain intact, ready for new data.

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

1. **`store_chat_summary(summary: str, original_date?: str)`** - Store a chat summary in the database
   - `original_date` (optional): ISO format date of the original conversation (used for recency weighting)
2. **`get_digital_me()`** - Retrieve the comprehensive digital me summary
3. **`answer_question(question: str, depth?: int, breadth?: int)`** - Answer questions using RAG retrieval
   - `depth` (optional, default 0): Number of parallel sub-agents (0-5)
   - `breadth` (optional, default 1): Retrieval multiplier for map-reduce (1-10, scans breadth×100 entries)

### Architecture

This server follows ZFC (Zero Framework Cognition) principles - pure orchestration that delegates ALL reasoning to external AI. The server provides:
- Database persistence (PostgreSQL/pgvector)
- LLM integration (OpenRouter - supports OpenAI, Anthropic, and other models)
- Vector embeddings (Cohere) for semantic search
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
