"""
LangGraph-based parallel multi-agent RAG system for Digital Me.

Implements parallel sub-agent execution where:
- depth=0: Single agent (simple retrieval + generation)
- depth>0: N parallel sub-agents with different prompt variants
- Each sub-agent independently retrieves and generates
- Central agent aggregates all sub-agent results

Map-reduce retrieval for large-scale queries:
- breadth=1 (default): Retrieve 100 entries
- breadth>1: Retrieve breadth*100 entries, process in chunks, aggregate

Based on: https://docs.langchain.com/oss/python/langgraph/agentic-rag
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import math

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

import numpy as np
from psycopg2.extras import RealDictCursor


@dataclass
class SubAgentResult:
    """Result from a parallel sub-agent."""
    agent_id: int
    variant_prompt: str
    answer: str
    chunks_retrieved: int
    confidence_score: float


class ParallelRAGState(MessagesState):
    """State for parallel multi-agent RAG system."""
    digital_me_summary: str
    depth: int
    query_variants: List[str]
    sub_agent_results: List[SubAgentResult]


def generate_query_variants(question: str, num_variants: int, llm_client) -> List[str]:
    """Generate N variants of the user query for parallel exploration.
    
    Args:
        question: Original user question
        num_variants: Number of variants to generate
        llm_client: LLM for variant generation
        
    Returns:
        List of query variants
    """
    if num_variants <= 1:
        return [question]
    
    variant_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a query expansion assistant. Generate {num_variants} different variants of the question that explore different aspects.

Each variant should:
1. Maintain the core intent
2. Explore different angles
3. Use different phrasing

Output ONLY a JSON array of strings, no explanation."""),
        ("human", "Generate {num_variants} variants:\n\n{question}")
    ])
    
    response = llm_client.invoke(
        variant_prompt.format_messages(num_variants=num_variants, question=question)
    )
    
    try:
        variants = json.loads(response.content)
        if isinstance(variants, list) and len(variants) >= num_variants:
            return variants[:num_variants]
    except:
        pass
    
    # Fallback: return original question repeated
    return [question] * num_variants


def run_subagent(
    agent_id: int,
    variant: str,
    digital_me_summary: str,
    llm_client,
    embeddings_client,
    db_connection,
    recency_weight: float = 0.3,
    recency_decay_days: float = 180.0
) -> SubAgentResult:
    """Run a single sub-agent: retrieve + generate with recency weighting.
    
    Args:
        agent_id: Sub-agent ID
        variant: Query variant
        digital_me_summary: Digital twin summary
        llm_client: LLM for generation
        embeddings_client: Embeddings for retrieval
        db_connection: Database connection
        recency_weight: Weight for recency boost (0.0-1.0, default 0.3)
        recency_decay_days: Decay half-life in days (default 180)
        
    Returns:
        SubAgentResult
    """
    # 1. Retrieve relevant chunks with recency weighting
    embedding = embeddings_client.embed_query(variant)
    embedding_array = np.array(embedding, dtype=np.float32)
    
    with db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        # Use recency-weighted retrieval if recency_weight > 0
        if recency_weight > 0:
            cur.execute("""
                SELECT 
                    summary_text, 
                    metadata,
                    (1 - (embedding <=> %s::vector)) as base_similarity,
                    CASE 
                        WHEN metadata->>'original_date' IS NOT NULL 
                        THEN EXTRACT(EPOCH FROM (NOW() - (metadata->>'original_date')::timestamp)) / 86400.0
                        ELSE NULL
                    END as days_old,
                    CASE 
                        WHEN metadata->>'original_date' IS NOT NULL 
                        THEN (1 - (embedding <=> %s::vector)) * (1 + %s * EXP(-(EXTRACT(EPOCH FROM (NOW() - (metadata->>'original_date')::timestamp)) / 86400.0) / %s))
                        ELSE (1 - (embedding <=> %s::vector))
                    END as boosted_similarity
                FROM chat_summaries
                WHERE embedding <=> %s::vector < 0.7
                ORDER BY boosted_similarity DESC
                LIMIT 100;
            """, (embedding_array, embedding_array, recency_weight, recency_decay_days, embedding_array, embedding_array))
        else:
            # Pure semantic search (no recency weighting)
            cur.execute("""
                SELECT summary_text, metadata,
                       (1 - (embedding <=> %s::vector)) as base_similarity,
                       NULL as days_old,
                       (1 - (embedding <=> %s::vector)) as boosted_similarity
                FROM chat_summaries
                WHERE embedding <=> %s::vector < 0.7
                ORDER BY embedding <=> %s::vector
                LIMIT 100;
            """, (embedding_array, embedding_array, embedding_array, embedding_array))
        
        chunks = cur.fetchall()
    
    # 2. Calculate confidence (use boosted_similarity if available)
    if chunks:
        similarities = [float(c.get('boosted_similarity', c.get('base_similarity', 0))) for c in chunks[:10]]
        confidence = sum(similarities) / len(similarities)
    else:
        confidence = 0.0
    
    # 3. Build context
    context_parts = []
    if digital_me_summary:
        context_parts.append(f"=== Digital Twin Profile ===\n{digital_me_summary}\n")
    
    if chunks:
        for i, chunk in enumerate(chunks[:20], 1):
            # Show both base and boosted similarity if available (convert Decimal to float)
            base_sim = float(chunk.get('base_similarity', chunk.get('similarity', 0)))
            boosted_sim = float(chunk.get('boosted_similarity', base_sim))
            days_old = chunk.get('days_old')
            
            if days_old is not None and recency_weight > 0:
                context_parts.append(
                    f"[{i}] (sim: {base_sim:.3f} → {boosted_sim:.3f}, {int(float(days_old))}d old)\n{chunk['summary_text']}"
                )
            else:
                context_parts.append(f"[{i}] (sim: {base_sim:.3f})\n{chunk['summary_text']}")
    
    context = "\n\n".join(context_parts) if context_parts else "No context found."
    
    # 4. Generate answer
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant answering questions about Andras based on his digital twin.

CRITICAL: ALL information refers to ANDRAS. "User" means ANDRAS.

Answer comprehensively using the provided context."""),
        ("human", """Context:
{context}

Question: {question}

Answer:""")
    ])
    
    response = llm_client.invoke(
        answer_prompt.format_messages(context=context, question=variant)
    )
    
    return SubAgentResult(
        agent_id=agent_id,
        variant_prompt=variant,
        answer=response.content,
        chunks_retrieved=len(chunks),
        confidence_score=confidence
    )


def retrieve_chunks_with_limit(
    query: str,
    embeddings_client,
    db_connection,
    limit: int = 100,
    recency_weight: float = 0.3,
    recency_decay_days: float = 180.0
) -> List[Dict[str, Any]]:
    """Retrieve chunks from database with specified limit.
    
    Args:
        query: Search query
        embeddings_client: Embeddings client
        db_connection: Database connection
        limit: Maximum number of chunks to retrieve
        recency_weight: Weight for recency boost
        recency_decay_days: Decay half-life in days
        
    Returns:
        List of chunk dictionaries
    """
    embedding = embeddings_client.embed_query(query)
    embedding_array = np.array(embedding, dtype=np.float32)
    
    with db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        if recency_weight > 0:
            cur.execute("""
                SELECT 
                    summary_text, 
                    metadata,
                    (1 - (embedding <=> %s::vector)) as base_similarity,
                    CASE 
                        WHEN metadata->>'original_date' IS NOT NULL 
                        THEN EXTRACT(EPOCH FROM (NOW() - (metadata->>'original_date')::timestamp)) / 86400.0
                        ELSE NULL
                    END as days_old,
                    CASE 
                        WHEN metadata->>'original_date' IS NOT NULL 
                        THEN (1 - (embedding <=> %s::vector)) * (1 + %s * EXP(-(EXTRACT(EPOCH FROM (NOW() - (metadata->>'original_date')::timestamp)) / 86400.0) / %s))
                        ELSE (1 - (embedding <=> %s::vector))
                    END as boosted_similarity
                FROM chat_summaries
                WHERE embedding <=> %s::vector < 0.7
                ORDER BY boosted_similarity DESC
                LIMIT %s;
            """, (embedding_array, embedding_array, recency_weight, recency_decay_days, embedding_array, embedding_array, limit))
        else:
            cur.execute("""
                SELECT summary_text, metadata,
                       (1 - (embedding <=> %s::vector)) as base_similarity,
                       NULL as days_old,
                       (1 - (embedding <=> %s::vector)) as boosted_similarity
                FROM chat_summaries
                WHERE embedding <=> %s::vector < 0.7
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (embedding_array, embedding_array, embedding_array, embedding_array, limit))
        
        return cur.fetchall()


def summarize_chunk_batch(
    batch_id: int,
    chunks: List[Dict[str, Any]],
    question: str,
    llm_client,
    recency_weight: float = 0.3
) -> Dict[str, Any]:
    """Summarize a batch of chunks into a sub-summary.
    
    Args:
        batch_id: Batch identifier
        chunks: List of chunks to summarize
        question: Original user question
        llm_client: LLM client
        recency_weight: Whether recency info is available
        
    Returns:
        Dictionary with batch_id, summary, and chunk count
    """
    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        base_sim = float(chunk.get('base_similarity', chunk.get('similarity', 0)))
        boosted_sim = float(chunk.get('boosted_similarity', base_sim))
        days_old = chunk.get('days_old')
        
        if days_old is not None and recency_weight > 0:
            context_parts.append(
                f"[{i}] (sim: {base_sim:.3f} → {boosted_sim:.3f}, {int(float(days_old))}d old)\n{chunk['summary_text']}"
            )
        else:
            context_parts.append(f"[{i}] (sim: {base_sim:.3f})\n{chunk['summary_text']}")
    
    context = "\n\n".join(context_parts)
    
    # Generate sub-summary
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant extracting relevant information about Andras from conversation summaries.

CRITICAL: ALL information refers to ANDRAS. "User" means ANDRAS.

Given a batch of conversation summaries and a question, extract and summarize ONLY the information relevant to answering the question. Be comprehensive but focused."""),
        ("human", """Question: {question}

Conversation Summaries (batch {batch_id}):
{context}

Extract relevant information for answering the question:""")
    ])
    
    response = llm_client.invoke(
        summarize_prompt.format_messages(question=question, batch_id=batch_id, context=context)
    )
    
    return {
        "batch_id": batch_id,
        "summary": response.content,
        "chunks_processed": len(chunks)
    }


def run_map_reduce_retrieval(
    question: str,
    llm_client,
    embeddings_client,
    db_connection,
    digital_me_summary: str = "",
    breadth: int = 1,
    recency_weight: float = 0.3,
    recency_decay_days: float = 180.0,
    chunk_size: int = 50
) -> Dict[str, Any]:
    """Run map-reduce retrieval for large-scale queries.
    
    Args:
        question: User's question
        llm_client: LLM client
        embeddings_client: Embeddings client
        db_connection: Database connection
        digital_me_summary: Digital twin summary
        breadth: Multiplier for retrieval (breadth * 100 entries)
        recency_weight: Weight for recency boost
        recency_decay_days: Decay half-life in days
        chunk_size: Number of entries per batch for map phase
        
    Returns:
        Dictionary with answer and metadata
    """
    # 1. Retrieve all chunks (breadth * 100)
    total_limit = breadth * 100
    all_chunks = retrieve_chunks_with_limit(
        query=question,
        embeddings_client=embeddings_client,
        db_connection=db_connection,
        limit=total_limit,
        recency_weight=recency_weight,
        recency_decay_days=recency_decay_days
    )
    
    if not all_chunks:
        return {
            "status": "success",
            "question": question,
            "answer": "No relevant information found in the database.",
            "execution_mode": "map_reduce",
            "breadth": breadth,
            "total_chunks_retrieved": 0,
            "batches_processed": 0
        }
    
    # 2. MAP phase: Chunk into batches and process in parallel
    num_batches = math.ceil(len(all_chunks) / chunk_size)
    batches = [
        all_chunks[i * chunk_size:(i + 1) * chunk_size]
        for i in range(num_batches)
    ]
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=min(num_batches, 10)) as executor:
        futures = [
            executor.submit(
                summarize_chunk_batch,
                i,
                batch,
                question,
                llm_client,
                recency_weight
            )
            for i, batch in enumerate(batches)
        ]
        batch_results = [f.result() for f in futures]
    
    # 3. REDUCE phase: Aggregate all sub-summaries
    if len(batch_results) == 1:
        # Single batch - use its summary as-is but format nicely
        final_answer = batch_results[0]["summary"]
    else:
        # Multiple batches - aggregate
        sub_summaries = []
        for r in batch_results:
            sub_summaries.append(
                f"=== Batch {r['batch_id'] + 1} ({r['chunks_processed']} conversations) ===\n{r['summary']}"
            )
        
        aggregated_context = "\n\n".join(sub_summaries)
        
        # Add digital_me context if available
        if digital_me_summary:
            aggregated_context = f"=== Digital Twin Profile ===\n{digital_me_summary}\n\n{aggregated_context}"
        
        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at synthesizing information about Andras from multiple sources.

Given extracted information from multiple batches of conversation summaries, create a comprehensive final answer.

CRITICAL: ALL information refers to ANDRAS. Be thorough - include ALL relevant details from the batches.

If the question asks about "all" or "every" instance of something, make sure to include everything found across all batches."""),
            ("human", """Question: {question}

Extracted Information from {num_batches} batches ({total_chunks} total conversations scanned):
{context}

Provide a comprehensive answer:""")
        ])
        
        response = llm_client.invoke(
            reduce_prompt.format_messages(
                question=question,
                num_batches=len(batch_results),
                total_chunks=len(all_chunks),
                context=aggregated_context
            )
        )
        final_answer = response.content
    
    return {
        "status": "success",
        "question": question,
        "answer": final_answer,
        "execution_mode": "map_reduce",
        "breadth": breadth,
        "total_chunks_retrieved": len(all_chunks),
        "batches_processed": len(batch_results),
        "chunks_per_batch": chunk_size,
        "recency_weight": recency_weight if recency_weight > 0 else None,
        "recency_decay_days": recency_decay_days if recency_weight > 0 else None
    }


def create_parallel_rag_graph(
    llm_client,
    embeddings_client,
    db_connection,
    digital_me_summary: str = "",
    depth: int = 0,
    recency_weight: float = 0.3,
    recency_decay_days: float = 180.0
):
    """Create LangGraph for parallel multi-agent RAG with recency weighting.
    
    Args:
        llm_client: LLM client
        embeddings_client: Embeddings client
        db_connection: Database connection
        digital_me_summary: Digital twin summary
        depth: Number of parallel sub-agents (0=single agent)
        recency_weight: Weight for recency boost (0.0-1.0)
        recency_decay_days: Decay half-life in days
        
    Returns:
        Compiled LangGraph
    """
    
    # Node 1: Generate query variants
    def generate_variants_node(state: ParallelRAGState):
        """Generate query variants for parallel sub-agents."""
        messages = state["messages"]
        user_question = messages[-1].content if messages else ""
        
        if depth == 0:
            # Single agent mode
            variants = [user_question]
        else:
            # Multi-agent mode
            variants = generate_query_variants(user_question, depth, llm_client)
        
        return {
            "query_variants": variants,
            "digital_me_summary": digital_me_summary,
            "depth": depth
        }
    
    # Node 2: Execute sub-agents in parallel
    def execute_subagents_parallel(state: ParallelRAGState):
        """Execute all sub-agents in parallel with recency weighting."""
        variants = state["query_variants"]
        
        # Run all sub-agents in parallel
        with ThreadPoolExecutor(max_workers=len(variants)) as executor:
            futures = [
                executor.submit(
                    run_subagent,
                    i,
                    variant,
                    state["digital_me_summary"],
                    llm_client,
                    embeddings_client,
                    db_connection,
                    recency_weight,
                    recency_decay_days
                )
                for i, variant in enumerate(variants)
            ]
            
            results = [f.result() for f in futures]
        
        return {"sub_agent_results": results}
    
    # Node 3: Aggregate results
    def aggregate_results_node(state: ParallelRAGState):
        """Aggregate sub-agent results into final answer."""
        messages = state["messages"]
        user_question = messages[-1].content if messages else ""
        results = state["sub_agent_results"]
        
        if len(results) == 1:
            # Single agent: return answer directly
            final_answer = results[0].answer
        else:
            # Multi-agent: aggregate using LLM
            answers_context = []
            for r in results:
                answers_context.append(
                    f"=== Answer {r.agent_id + 1} (confidence: {float(r.confidence_score):.2f}) ===\n"
                    f"Variant: {r.variant_prompt}\n"
                    f"Answer: {r.answer}\n"
                    f"Chunks: {r.chunks_retrieved}"
                )
            
            context = "\n\n".join(answers_context)
            
            aggregation_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at synthesizing multiple perspectives into a comprehensive answer.

Given multiple answers to variants of the same question, create a final answer that:
1. Synthesizes information from all answers
2. Weights answers by confidence scores
3. Resolves conflicts
4. Provides the most complete and accurate response

Focus on information about ANDRAS."""),
                ("human", """Original Question: {question}

Multiple Answers:
{context}

Synthesize into a single comprehensive answer:""")
            ])
            
            response = llm_client.invoke(
                aggregation_prompt.format_messages(question=user_question, context=context)
            )
            
            final_answer = response.content
        
        return {"messages": [AIMessage(content=final_answer)]}
    
    # Build the graph
    workflow = StateGraph(ParallelRAGState)
    
    # Add nodes
    workflow.add_node("generate_variants", generate_variants_node)
    workflow.add_node("execute_subagents", execute_subagents_parallel)
    workflow.add_node("aggregate_results", aggregate_results_node)
    
    # Add edges
    workflow.add_edge(START, "generate_variants")
    workflow.add_edge("generate_variants", "execute_subagents")
    workflow.add_edge("execute_subagents", "aggregate_results")
    workflow.add_edge("aggregate_results", END)
    
    return workflow.compile()


def run_agentic_rag(
    question: str,
    llm_client,
    embeddings_client,
    db_connection,
    digital_me_summary: str = "",
    depth: int = 0,
    recency_weight: float = 0.3,
    recency_decay_days: float = 180.0
) -> Dict[str, Any]:
    """Run parallel multi-agent RAG system with recency weighting.
    
    Args:
        question: User's question
        llm_client: LLM client
        embeddings_client: Embeddings client
        db_connection: Database connection
        digital_me_summary: Digital twin summary
        depth: Number of parallel sub-agents (0=single)
        recency_weight: Weight for recency boost (0.0-1.0, default 0.3)
        recency_decay_days: Decay half-life in days (default 180)
        
    Returns:
        Dictionary with answer and metadata
    """
    # Create and run graph
    graph = create_parallel_rag_graph(
        llm_client=llm_client,
        embeddings_client=embeddings_client,
        db_connection=db_connection,
        digital_me_summary=digital_me_summary,
        depth=max(1, depth),  # At least 1 agent
        recency_weight=recency_weight,
        recency_decay_days=recency_decay_days
    )
    
    result = graph.invoke({
        "messages": [HumanMessage(content=question)],
        "digital_me_summary": digital_me_summary,
        "depth": depth,
        "query_variants": [],
        "sub_agent_results": []
    })
    
    # Extract results
    final_message = result["messages"][-1]
    answer = final_message.content
    sub_results = result.get("sub_agent_results", [])
    
    # Build response
    return {
        "status": "success",
        "question": question,
        "answer": answer,
        "execution_mode": "langgraph_parallel_agents",
        "depth": depth,
        "num_agents": len(sub_results),
        "total_chunks": sum(r.chunks_retrieved for r in sub_results),
        "avg_confidence": sum(float(r.confidence_score) for r in sub_results) / len(sub_results) if sub_results else 0,
        "variants": [r.variant_prompt for r in sub_results],
        "recency_weight": recency_weight if recency_weight > 0 else None,
        "recency_decay_days": recency_decay_days if recency_weight > 0 else None
    }

