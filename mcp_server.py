"""
MCP Server for Digital Me - Digital Twin of Andras

This server provides tools for managing a digital twin of Andras, an individual:
1. Storing chat summaries and conversations about Andras in PostgreSQL/pgvector for RAG
2. Maintaining a comprehensive "digital me" summary record representing Andras's knowledge, interests, preferences, and context
3. Answering questions about Andras using semantic search over stored conversations

All reasoning is delegated to external AI (ZFC-compliant).
"""

import os
import json
import re
import hashlib
from typing import Optional
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv
import fastmcp
from fastmcp import FastMCP
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from pgvector.psycopg2 import register_vector

from ai_providers import CohereEmbeddingsClient, OpenRouterChat

# Load environment variables
load_dotenv()

# Initialize MCP server
mcp = FastMCP("Digital Me MCP Server - Digital Twin of Andras")

# Database connection pool
db_pool: Optional[ThreadedConnectionPool] = None

# LLM and embeddings clients
llm = None
embeddings = None


def get_embedding_dimension() -> int:
    """Determine embedding dimensionality based on configuration."""
    env_dim = os.getenv("EMBEDDING_DIMENSION")
    if env_dim:
        try:
            value = int(env_dim)
            if value <= 0:
                raise ValueError
            return value
        except ValueError as exc:
            raise ValueError("EMBEDDING_DIMENSION must be a positive integer") from exc
    provider = os.getenv("EMBEDDING_PROVIDER", "cohere-v2").lower()
    if provider in {"cohere", "cohere-v2"}:
        model_name = os.getenv("EMBEDDING_MODEL", "").lower()
        if "v3" in model_name or "small" in model_name:
            return 1024
        return 4096
    return 1536


def get_llm():
    """Initialize LLM based on .env configuration (for backwards compatibility)"""
    global llm
    if llm is not None:
        return llm
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")
    base_url = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")
    if not model:
        raise ValueError("OPENROUTER_MODEL not found in .env")
    llm = OpenRouterChat(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
    )
    
    return llm


def get_central_agent_llm():
    """Initialize LLM for central agent."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_CENTRAL_AGENT_MODEL", os.getenv("OPENROUTER_MODEL"))
    base_url = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1")
    temperature = float(os.getenv("CENTRAL_AGENT_TEMPERATURE", "0.7"))
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")
    if not model:
        raise ValueError("OPENROUTER_CENTRAL_AGENT_MODEL or OPENROUTER_MODEL not found in .env")
    
    return OpenRouterChat(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
    )


def get_subagent_llm():
    """Initialize LLM for sub-agents."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_SUBAGENT_MODEL", os.getenv("OPENROUTER_MODEL"))
    base_url = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1")
    temperature = float(os.getenv("SUBAGENT_TEMPERATURE", "0.7"))
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")
    if not model:
        raise ValueError("OPENROUTER_SUBAGENT_MODEL or OPENROUTER_MODEL not found in .env")
    
    return OpenRouterChat(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
    )


def get_embeddings():
    """Initialize embeddings model based on .env configuration"""
    global embeddings
    if embeddings is not None:
        return embeddings
    provider = os.getenv("EMBEDDING_PROVIDER", "cohere-v2").lower()
    if provider not in {"cohere", "cohere-v2"}:
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER: {provider}. Use 'cohere' or 'cohere-v2'"
        )
    api_key = os.getenv("COHERE_API_KEY")
    model = os.getenv("EMBEDDING_MODEL", "embed-english-v2.0")
    truncate = os.getenv("COHERE_TRUNCATE", "END") or None
    if not api_key:
        raise ValueError("COHERE_API_KEY not found in .env")
    embeddings = CohereEmbeddingsClient(api_key=api_key, model=model, truncate=truncate)
    
    return embeddings


def get_db_connection(register_vector_type: bool = True):
    """Get a database connection from the pool"""
    global db_pool
    if db_pool is None:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL not found in .env")
        db_pool = ThreadedConnectionPool(1, 5, db_url)
    conn = db_pool.getconn()
    # Register pgvector adapter (only if extension exists)
    if register_vector_type:
        try:
            register_vector(conn)
        except psycopg2.ProgrammingError:
            # Extension not created yet, will be created in init_database
            pass
    return conn


def return_db_connection(conn):
    """Return a connection to the pool"""
    global db_pool
    if db_pool:
        db_pool.putconn(conn)


def init_database():
    """Initialize database schema if it doesn't exist"""
    # Get connection without registering vector (extension doesn't exist yet)
    conn = get_db_connection(register_vector_type=False)
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            
            # Now register vector type after extension is created
            register_vector(conn)
            
            embedding_dimension = get_embedding_dimension()
            
            # Create chat_summaries table
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chat_summaries (
                    id SERIAL PRIMARY KEY,
                    summary_text TEXT NOT NULL,
                    embedding vector({embedding_dimension}),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            
            # Create digital_me table (single record)
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS digital_me (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    summary_text TEXT NOT NULL DEFAULT '',
                    embedding vector({embedding_dimension}),
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT single_record CHECK (id = 1)
                );
                """
            )
            
            # Validate embedding dimensions if tables already exist
            cur.execute(
                """
                SELECT atttypmod AS dimension
                FROM pg_attribute
                WHERE attrelid = 'chat_summaries'::regclass
                  AND attname = 'embedding';
                """
            )
            chat_dimension = cur.fetchone()
            if (
                chat_dimension
                and chat_dimension[0] not in (None, -1, embedding_dimension)
            ):
                raise ValueError(
                    f"chat_summaries.embedding has dimension {chat_dimension[0]} but configuration expects {embedding_dimension}. "
                    "Please migrate or adjust EMBEDDING_DIMENSION."
                )
            
            cur.execute(
                """
                SELECT atttypmod AS dimension
                FROM pg_attribute
                WHERE attrelid = 'digital_me'::regclass
                  AND attname = 'embedding';
                """
            )
            digital_me_dimension = cur.fetchone()
            if (
                digital_me_dimension
                and digital_me_dimension[0] not in (None, -1, embedding_dimension)
            ):
                raise ValueError(
                    f"digital_me.embedding has dimension {digital_me_dimension[0]} but configuration expects {embedding_dimension}. "
                    "Please migrate or adjust EMBEDDING_DIMENSION."
                )
            
            # Create index for vector similarity search
            if embedding_dimension <= 2000:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS chat_summaries_embedding_idx 
                    ON chat_summaries USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """
                )
            
            # Initialize digital_me if it doesn't exist
            cur.execute("""
                INSERT INTO digital_me (id, summary_text)
                VALUES (1, '')
                ON CONFLICT (id) DO NOTHING;
            """)
            
            conn.commit()
    finally:
        return_db_connection(conn)


@mcp.tool()
def store_chat_summary(summary: str, original_date: Optional[str] = None) -> dict:
    """
    Store a chat summary or conversation about Andras in the database for RAG retrieval.
    
    This tool stores conversations and summaries related to Andras, building his digital twin:
    1. Uses an LLM to process the summary into an optimal RAG format
    2. Generates embeddings for semantic vector search
    3. Stores the processed summary in PostgreSQL/pgvector
    4. Updates Andras's "digital me" record with new information about his knowledge, interests, preferences, and context
    
    Args:
        summary: A summary or full text of a chat conversation about Andras
        original_date: ISO format date string of the original conversation (optional)
        
    Returns:
        A dictionary with status and details about the stored summary, including chat_id and created_at timestamp
    """
    try:
        # Initialize database if needed
        init_database()
        
        # Get LLM for processing
        llm_client = get_llm()
        embeddings_client = get_embeddings()
        
        # Step 1: Process summary with LLM for RAG optimization
        process_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a RAG (Retrieval Augmented Generation) optimization assistant.
Your task is to process a chat summary or conversation about Andras into an optimal format for retrieval.

Transform the summary into a clear, structured text that:
1. Preserves all important information about Andras
2. Uses clear, searchable language
3. Extracts key facts, interests, preferences, expertise areas, projects, goals, and context about Andras
4. Is optimized for semantic search and retrieval

Output only the processed summary text, nothing else."""),
            ("human", "Process this chat summary for RAG storage:\n\n{summary}")
        ])
        
        processed_summary = llm_client.invoke(
            process_prompt.format_messages(summary=summary)
        ).content
        
        # Step 2: Generate embedding
        embedding_vector = embeddings_client.embed_query(processed_summary)
        
        # Step 3: Store in database
        # Compute hash of original summary for duplicate detection
        summary_hash = hashlib.sha256(summary.encode('utf-8')).hexdigest()
        
        conn = get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if summary with this hash already exists
                cur.execute("""
                    SELECT id FROM chat_summaries 
                    WHERE metadata->>'summary_hash' = %s
                    LIMIT 1;
                """, (summary_hash,))
                existing = cur.fetchone()
                
                if existing:
                    return {
                        "status": "skipped",
                        "message": "Chat summary already exists in database",
                        "chat_id": existing['id'],
                        "reason": "duplicate"
                    }
                
                # Convert embedding to numpy array for pgvector
                embedding_array = np.array(embedding_vector, dtype=np.float32)
                metadata = {
                    "original_summary": summary,
                    "summary_hash": summary_hash
                }
                # Add original_date to metadata if provided
                if original_date:
                    metadata["original_date"] = original_date
                cur.execute("""
                    INSERT INTO chat_summaries (summary_text, embedding, metadata)
                    VALUES (%s, %s, %s)
                    RETURNING id, created_at;
                """, (processed_summary, embedding_array, json.dumps(metadata)))
                
                result = cur.fetchone()
                chat_id = result['id']
                created_at = result['created_at']
                
                conn.commit()
        finally:
            return_db_connection(conn)
        
        # Step 4: Update digital_me record
        update_digital_me(processed_summary)
        
        result = {
            "status": "success",
            "message": "Chat summary stored successfully",
            "chat_id": chat_id,
            "created_at": created_at.isoformat(),
            "processed_summary_length": len(processed_summary)
        }
        if original_date:
            result["original_date"] = original_date
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to store chat summary: {str(e)}"
        }


def update_digital_me(new_summary: str):
    """Update the digital_me record with new information"""
    try:
        llm_client = get_llm()
        conn = get_db_connection()
        
        try:
            # Get current digital_me record
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT summary_text FROM digital_me WHERE id = 1;")
                result = cur.fetchone()
                current_summary = result['summary_text'] if result else ""
            
            # Use LLM to merge new information into existing summary
            merge_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are maintaining a comprehensive digital twin summary about Andras.
Your task is to update this summary by incorporating new information from a chat summary or conversation about Andras.

Guidelines:
1. Preserve all existing important information about Andras
2. Add new facts, interests, preferences, and context about Andras
3. Update or refine existing information if the new data provides more detail about Andras
4. Remove outdated information if contradicted by new data about Andras
5. Keep the summary comprehensive but well-organized
6. Focus on Andras's facts, interests, preferences, skills, goals, projects, expertise areas, and personality traits

Output only the updated summary text, nothing else."""),
                ("human", """Current digital me summary:
{current_summary}

New information from chat:
{new_summary}

Update the digital me summary by incorporating the new information:""")
            ])
            
            updated_summary = llm_client.invoke(
                merge_prompt.format_messages(
                    current_summary=current_summary,
                    new_summary=new_summary
                )
            ).content
            
            # Generate embedding for updated summary
            embeddings_client = get_embeddings()
            embedding_vector = embeddings_client.embed_query(updated_summary)
            
            # Update database
            with conn.cursor() as cur:
                # Convert embedding to numpy array for pgvector
                embedding_array = np.array(embedding_vector, dtype=np.float32)
                cur.execute("""
                    UPDATE digital_me
                    SET summary_text = %s, embedding = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1;
                """, (updated_summary, embedding_array))
                
                conn.commit()
        finally:
            return_db_connection(conn)
            
    except Exception as e:
        # Log error but don't fail the main operation
        print(f"Warning: Failed to update digital_me: {str(e)}")


@mcp.tool()
def get_digital_me() -> dict:
    """
    Retrieve Andras's digital twin summary record.
    
    This returns the comprehensive digital twin summary of Andras, which includes:
    - His knowledge, interests, and expertise areas
    - His preferences and opinions
    - His goals, projects, and context
    - His personality traits and communication style
    
    This summary is continuously updated as new chat summaries and conversations about Andras are stored.
    
    Returns:
        A dictionary containing Andras's digital twin summary and the last updated timestamp
    """
    try:
        init_database()
        conn = get_db_connection()
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT summary_text, updated_at
                    FROM digital_me
                    WHERE id = 1;
                """)
                
                result = cur.fetchone()
                
                if result:
                    return {
                        "status": "success",
                        "summary": result['summary_text'],
                        "updated_at": result['updated_at'].isoformat() if result['updated_at'] else None
                    }
                else:
                    return {
                        "status": "success",
                        "summary": "",
                        "updated_at": None,
                        "message": "Digital me record is empty"
                    }
        finally:
            return_db_connection(conn)
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to retrieve digital me: {str(e)}"
        }


@mcp.tool()
def answer_question(question: str, depth: int = 0) -> dict:
    """
    Answer a question about Andras using semantic search over his digital twin data.
    
    This tool queries Andras's digital twin using single-agent or multi-agent RAG based on depth:
    - depth=0 (basic): Single RAG query with central agent only (fast, cost-effective)
    - depth=1-5 (advanced): Multi-agent RAG with N parallel sub-agents exploring different query variants
    
    Higher depth provides more comprehensive answers but costs more tokens and takes longer.
    
    Args:
        question: A question about Andras (e.g., "What domains of quantum research is Andras focused on?", "What are Andras's interests?", "What projects is Andras working on?")
        depth: Research depth (0=basic single-agent, 1-5=multi-agent with N variants). Default is 0.
        
    Returns:
        A dictionary containing the answer about Andras and the number of context sources used
    """
    try:
        init_database()
        
        # Validate depth parameter
        if depth < 0:
            depth = 0
        elif depth > 5:
            depth = 5
        
        # depth=0 means single-agent, depth>0 means multi-agent with that many variants
        if depth == 0:
            return _answer_question_single_agent(question)
        else:
            return _answer_question_multi_agent(question, num_variants=depth)
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": f"Failed to answer question: {str(e)}",
            "traceback": traceback.format_exc()
        }


def _answer_question_multi_agent(question: str, num_variants: int = 3) -> dict:
    """Answer question using parallel LangGraph agentic RAG system.
    
    Args:
        question: User's question
        num_variants: Number of parallel sub-agents (depth parameter)
    """
    from agents import run_agentic_rag
    
    # Get digital_me summary
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT summary_text FROM digital_me WHERE id = 1;")
            digital_me_result = cur.fetchone()
            digital_me_summary = digital_me_result['summary_text'] if digital_me_result else ""
    finally:
        pass  # Don't return connection yet, agents will use it
    
    # Get clients
    llm_client = get_central_agent_llm()
    embeddings_client = get_embeddings()
    
    # Get recency configuration
    recency_weight = float(os.getenv("RECENCY_WEIGHT", "0.3"))
    recency_decay_days = float(os.getenv("RECENCY_DECAY_DAYS", "180"))
    
    # Run parallel LangGraph agentic RAG
    result = run_agentic_rag(
        question=question,
        llm_client=llm_client,
        embeddings_client=embeddings_client,
        db_connection=conn,
        digital_me_summary=digital_me_summary,
        depth=num_variants,
        recency_weight=recency_weight,
        recency_decay_days=recency_decay_days
    )
    
    # Return connection
    return_db_connection(conn)
    
    return result


def _answer_question_single_agent(question: str) -> dict:
    """Answer question using traditional single-agent approach."""
    llm_client = get_llm()
    embeddings_client = get_embeddings()
    
    # Generate embedding for the question
    question_embedding = embeddings_client.embed_query(question)
    
    # Extract potential keywords from question for hybrid search
    # Focus on specific topic keywords, not names or user-related terms
    keywords = []
    # Extract quoted phrases
    quoted = re.findall(r'"([^"]+)"', question)
    keywords.extend(quoted)
    # Extract lowercase words that might be important (remove common stop words AND user-related terms)
    stop_words = {'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'about', 'what', 'where', 'when', 'who', 'why', 'how', 'does', 'do', 'did', 'has', 'have', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'user', 'users', 'person', 'people', 'inquire', 'inquired', 'ask', 'asked', 'tell', 'told', 'sort', 'kind', 'type'}
    words = re.findall(r'\b[a-z]+\b', question.lower())
    keywords.extend([w for w in words if w not in stop_words and len(w) > 3])
    keywords = list(set(keywords))  # Remove duplicates
    
    # Search for relevant chat summaries
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Convert embedding to numpy array for pgvector
            question_embedding_array = np.array(question_embedding, dtype=np.float32)
            
            # HYBRID SEARCH: Combine semantic search with keyword search
            semantic_summaries = []
            keyword_summaries = []
            
            # 1. Semantic search with relaxed threshold
            cur.execute("""
                SELECT summary_text, metadata,
                       (1 - (embedding <=> %s::vector)) as similarity
                FROM chat_summaries
                WHERE embedding <=> %s::vector < 0.7
                ORDER BY embedding <=> %s::vector
                LIMIT 100;
            """, (question_embedding_array, question_embedding_array, question_embedding_array))
            semantic_summaries = cur.fetchall()
            
            # 2. Keyword search for exact matches (prioritize these)
            if keywords:
                # Build keyword search query
                keyword_conditions = []
                keyword_params = []
                for keyword in keywords:
                    keyword_conditions.append("summary_text ILIKE %s")
                    keyword_params.append(f"%{keyword}%")
                
                keyword_query = f"""
                    SELECT summary_text, metadata,
                           1.0 as similarity
                    FROM chat_summaries
                    WHERE {' OR '.join(keyword_conditions)}
                    LIMIT 100;
                """
                cur.execute(keyword_query, tuple(keyword_params))
                keyword_summaries = cur.fetchall()
            
            # Combine and deduplicate results (prioritize keyword matches)
            seen_text_hashes = set()
            relevant_summaries = []
            
            # First add keyword matches (higher priority, similarity = 1.0)
            for summary in keyword_summaries:
                text_hash = hashlib.md5(summary['summary_text'].encode()).hexdigest()
                if text_hash not in seen_text_hashes:
                    relevant_summaries.append(summary)
                    seen_text_hashes.add(text_hash)
            
            # Then add semantic matches that aren't already included
            for summary in semantic_summaries:
                text_hash = hashlib.md5(summary['summary_text'].encode()).hexdigest()
                if text_hash not in seen_text_hashes:
                    relevant_summaries.append(summary)
                    seen_text_hashes.add(text_hash)
            
            # Limit to top 100 total
            relevant_summaries = relevant_summaries[:100]
            
            # Get digital_me summary
            cur.execute("SELECT summary_text FROM digital_me WHERE id = 1;")
            digital_me_result = cur.fetchone()
            digital_me_summary = digital_me_result['summary_text'] if digital_me_result else ""
    finally:
        return_db_connection(conn)
    
    # Prepare context for LLM
    context_parts = []
    if digital_me_summary:
        context_parts.append(f"Digital Me Summary (Andras's comprehensive profile):\n{digital_me_summary}")
    
    if relevant_summaries:
        # Count keyword vs semantic matches
        keyword_count = sum(1 for s in relevant_summaries if s.get('similarity', 0) >= 0.99)
        semantic_count = len(relevant_summaries) - keyword_count
        
        context_parts.append(f"\nRelevant Chat Summaries ({len(relevant_summaries)} found: {keyword_count} keyword matches, {semantic_count} semantic matches):")
        for i, summary in enumerate(relevant_summaries, 1):
            similarity = summary.get('similarity', 0)
            text = summary['summary_text']
            match_type = "KEYWORD" if similarity >= 0.99 else "SEMANTIC"
            # Truncate very long summaries to avoid token limits
            if len(text) > 2000:
                text = text[:2000] + "... [truncated]"
            context_parts.append(f"\n[{i}] [{match_type}] (similarity: {similarity:.3f})\n{text}")
    else:
        context_parts.append("\nNo relevant chat summaries found via hybrid search (semantic + keyword).")
    
    context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
    
    # Use LLM to answer the question
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant that answers questions about Andras (also known as András Ferenczi) based on his chat history and digital twin profile.

CRITICAL IDENTITY MAPPING:
- ANY reference to "User", "user", "the user", or "user profile" in the context refers to ANDRAS
- ANY conversation, question, or inquiry in the context was made BY ANDRAS
- ANY interest, preference, or activity mentioned in the context belongs to ANDRAS
- This is Andras's personal digital twin database - EVERY piece of information is about HIM

IMPORTANT: ALL conversations and summaries in the provided context are Andras's own conversations. When you see:
- "User asked about X" → Andras asked about X
- "User is interested in Y" → Andras is interested in Y  
- "User profile" → Andras's profile
- "User context" → Andras's context

Carefully review ALL the provided context from Andras's stored conversations and digital twin summary. Extract and synthesize information from ALL relevant sources to answer the question comprehensively.

If the question asks about specific topics, places, interests, or activities (like "sciatica", "Naples", "quantum", "interests", "projects", etc.), search through ALL the provided context for mentions of those topics, even if they appear in multiple summaries. If a conversation mentions places to visit, activities, health concerns, interests, or preferences, those are Andras's interests and preferences.

Use the provided context to answer the question accurately and comprehensively. Include specific details, examples, and nuances from the context. 

Be factual and thorough. Focus on what the context tells you about Andras specifically. Do NOT dismiss conversations as "belonging to a different user" or say "there is no record" when the context clearly contains relevant information about "User" - that User IS Andras."""),
        ("human", """Context:
{context}

Question: {question}

Answer:""")
    ])
    
    answer = llm_client.invoke(
        answer_prompt.format_messages(context=context, question=question)
    ).content
    
    # Calculate average similarity and keyword match count for debugging
    avg_similarity = None
    keyword_match_count = 0
    if relevant_summaries:
        similarities = [float(s.get('similarity', 0)) for s in relevant_summaries if 'similarity' in s]
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
        keyword_match_count = sum(1 for s in relevant_summaries if float(s.get('similarity', 0)) >= 0.99)
    
    return {
        "status": "success",
        "question": question,
        "answer": answer,
        "context_sources": len(relevant_summaries) + (1 if digital_me_summary else 0),
        "summaries_found": len(relevant_summaries),
        "keyword_matches": keyword_match_count,
        "semantic_matches": len(relevant_summaries) - keyword_match_count,
        "avg_similarity": round(avg_similarity, 3) if avg_similarity is not None else None,
        "execution_mode": "single_agent",
        "depth": 0
    }


if __name__ == "__main__":
    # Initialize database on startup
    init_database()
    
    # Run the MCP server
    host = os.getenv("MCP_SERVER_HOST", fastmcp.settings.host)
    port = int(os.getenv("MCP_SERVER_PORT", fastmcp.settings.port))
    mcp.run(transport="streamable-http", host=host, port=port)

