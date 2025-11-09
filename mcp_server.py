"""
MCP Server for Digital Me - Intelligent Chat Summary Storage with RAG

This server provides tools for:
1. Storing chat summaries in PostgreSQL/pgvector for RAG
2. Maintaining a "digital me" summary record
3. Answering ad-hoc questions about the user

All reasoning is delegated to external AI (ZFC-compliant).
"""

import os
import json
from typing import Optional
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from pgvector.psycopg2 import register_vector

# Load environment variables
load_dotenv()

# Initialize MCP server
mcp = FastMCP("Digital Me MCP Server")

# Database connection pool
db_pool: Optional[ThreadedConnectionPool] = None

# LLM and embeddings clients
llm = None
embeddings = None


def get_llm():
    """Initialize LLM based on .env configuration"""
    global llm
    if llm is not None:
        return llm
    
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.7)
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env")
        llm = ChatAnthropic(model=model, api_key=api_key, temperature=0.7)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}. Use 'openai' or 'anthropic'")
    
    return llm


def get_embeddings():
    """Initialize embeddings model based on .env configuration"""
    global embeddings
    if embeddings is not None:
        return embeddings
    
    provider = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)
    else:
        raise ValueError(f"Unsupported EMBEDDINGS_PROVIDER: {provider}. Use 'openai'")
    
    return embeddings


def get_db_connection():
    """Get a database connection from the pool"""
    global db_pool
    if db_pool is None:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL not found in .env")
        db_pool = ThreadedConnectionPool(1, 5, db_url)
    conn = db_pool.getconn()
    # Register pgvector adapter
    register_vector(conn)
    return conn


def return_db_connection(conn):
    """Return a connection to the pool"""
    global db_pool
    if db_pool:
        db_pool.putconn(conn)


def init_database():
    """Initialize database schema if it doesn't exist"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create chat_summaries table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_summaries (
                    id SERIAL PRIMARY KEY,
                    summary_text TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create digital_me table (single record)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS digital_me (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    summary_text TEXT NOT NULL DEFAULT '',
                    embedding vector(1536),
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT single_record CHECK (id = 1)
                );
            """)
            
            # Create index for vector similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chat_summaries_embedding_idx 
                ON chat_summaries USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
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
def store_chat_summary(summary: str) -> dict:
    """
    Store a chat summary in the database for RAG retrieval.
    
    This tool:
    1. Uses an LLM to process the summary into an appropriate RAG format
    2. Generates embeddings for vector search
    3. Stores the processed summary in PostgreSQL/pgvector
    4. Updates the "digital me" record with new information about the user
    
    Args:
        summary: A summary of a chat conversation
        
    Returns:
        A dictionary with status and details about the stored summary
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
Your task is to process a chat summary into an optimal format for retrieval.

Transform the summary into a clear, structured text that:
1. Preserves all important information about the user
2. Uses clear, searchable language
3. Extracts key facts, interests, preferences, and context
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
        conn = get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Convert embedding to numpy array for pgvector
                embedding_array = np.array(embedding_vector, dtype=np.float32)
                cur.execute("""
                    INSERT INTO chat_summaries (summary_text, embedding, metadata)
                    VALUES (%s, %s, %s)
                    RETURNING id, created_at;
                """, (processed_summary, embedding_array, json.dumps({"original_summary": summary})))
                
                result = cur.fetchone()
                chat_id = result['id']
                created_at = result['created_at']
                
                conn.commit()
        finally:
            return_db_connection(conn)
        
        # Step 4: Update digital_me record
        update_digital_me(processed_summary)
        
        return {
            "status": "success",
            "message": "Chat summary stored successfully",
            "chat_id": chat_id,
            "created_at": created_at.isoformat(),
            "processed_summary_length": len(processed_summary)
        }
        
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
                ("system", """You are maintaining a comprehensive summary about a person (the "digital me").
Your task is to update this summary by incorporating new information from a chat summary.

Guidelines:
1. Preserve all existing important information
2. Add new facts, interests, preferences, and context
3. Update or refine existing information if the new data provides more detail
4. Remove outdated information if contradicted by new data
5. Keep the summary comprehensive but well-organized
6. Focus on facts, interests, preferences, skills, goals, and personality traits

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
    Retrieve the "digital me" summary record.
    
    This returns the comprehensive summary about the user that has been
    continuously updated as new chat summaries are added.
    
    Returns:
        A dictionary containing the digital me summary and metadata
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
def answer_question(question: str) -> dict:
    """
    Answer an ad-hoc question about the user using RAG retrieval.
    
    This tool:
    1. Searches the chat summaries and digital_me record using semantic search
    2. Retrieves relevant context
    3. Uses an LLM to answer the question based on the retrieved context
    
    Args:
        question: A question about the user (e.g., "Is Andras interested in quantum computing?")
        
    Returns:
        A dictionary containing the answer and relevant context
    """
    try:
        init_database()
        llm_client = get_llm()
        embeddings_client = get_embeddings()
        
        # Generate embedding for the question
        question_embedding = embeddings_client.embed_query(question)
        
        # Search for relevant chat summaries
        conn = get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Search chat summaries using cosine similarity
                # Convert embedding to numpy array for pgvector
                question_embedding_array = np.array(question_embedding, dtype=np.float32)
                cur.execute("""
                    SELECT summary_text, metadata,
                           (1 - (embedding <=> %s::vector)) as similarity
                    FROM chat_summaries
                    WHERE embedding <=> %s::vector < 0.5
                    ORDER BY embedding <=> %s::vector
                    LIMIT 5;
                """, (question_embedding_array, question_embedding_array, question_embedding_array))
                
                relevant_summaries = cur.fetchall()
                
                # Get digital_me summary
                cur.execute("SELECT summary_text FROM digital_me WHERE id = 1;")
                digital_me_result = cur.fetchone()
                digital_me_summary = digital_me_result['summary_text'] if digital_me_result else ""
        finally:
            return_db_connection(conn)
        
        # Prepare context for LLM
        context_parts = []
        if digital_me_summary:
            context_parts.append(f"Digital Me Summary:\n{digital_me_summary}")
        
        if relevant_summaries:
            context_parts.append("\nRelevant Chat Summaries:")
            for summary in relevant_summaries:
                context_parts.append(f"- {summary['summary_text']}")
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        
        # Use LLM to answer the question
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant that answers questions about a person based on their chat history and digital profile.

Use the provided context to answer the question accurately. If the context doesn't contain enough information to answer definitively, say so.

Be concise and factual."""),
            ("human", """Context:
{context}

Question: {question}

Answer:""")
        ])
        
        answer = llm_client.invoke(
            answer_prompt.format_messages(context=context, question=question)
        ).content
        
        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "context_sources": len(relevant_summaries) + (1 if digital_me_summary else 0)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to answer question: {str(e)}"
        }


if __name__ == "__main__":
    # Initialize database on startup
    init_database()
    
    # Run the MCP server
    mcp.run()

