#!/usr/bin/env python3
"""
Extract all conversations and summaries from the database to separate files.

This script queries the PostgreSQL database and extracts:
1. Original conversations (from metadata->>'original_summary') to all_conversations.txt
2. Processed summaries (from summary_text) to all_summaries.txt

Both files are written with proper UTF-8 encoding.

Usage:
    python scripts/extract_from_db.py [--conversations all_conversations.txt] [--summaries all_summaries.txt]
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DEFAULT_CONVERSATIONS_FILE = "all_conversations.txt"
DEFAULT_SUMMARIES_FILE = "all_summaries.txt"


def get_db_connection():
    """Get a database connection from DATABASE_URL."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found in .env file")
    return psycopg2.connect(database_url)


def extract_all_conversations(output_file: Path):
    """Extract all original conversations from database and write to file."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Query all chat summaries with original_summary from metadata
            cur.execute("""
                SELECT 
                    id,
                    metadata->>'original_summary' as original_summary,
                    created_at
                FROM chat_summaries
                WHERE metadata->>'original_summary' IS NOT NULL
                  AND metadata->>'original_summary' != ''
                ORDER BY created_at ASC, id ASC;
            """)
            
            conversations = cur.fetchall()
            
            if not conversations:
                print("No conversations found in database")
                return 0
            
            print(f"Found {len(conversations)} conversations in database")
            print(f"Writing to: {output_file}")
            
            # Write all conversations to file with proper UTF-8 encoding
            with open(output_file, "w", encoding="utf-8", errors="replace") as f:
                for i, conv in enumerate(conversations, 1):
                    original_summary = conv['original_summary']
                    if not original_summary:
                        continue
                    
                    f.write(f"{'='*80}\n")
                    f.write(f"Conversation {i}/{len(conversations)}")
                    f.write(f" (ID: {conv['id']}, Created: {conv['created_at']})\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(original_summary)
                    f.write("\n\n")
            
            file_size = output_file.stat().st_size
            print(f"✅ Successfully wrote {len(conversations)} conversations to {output_file}")
            print(f"   File size: {file_size:,} bytes")
            return len(conversations)
            
    finally:
        conn.close()


def extract_all_summaries(output_file: Path):
    """Extract all processed summaries from database and write to file."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Query all chat summaries with processed summary_text
            cur.execute("""
                SELECT 
                    id,
                    summary_text,
                    created_at
                FROM chat_summaries
                WHERE summary_text IS NOT NULL
                  AND summary_text != ''
                ORDER BY created_at ASC, id ASC;
            """)
            
            summaries = cur.fetchall()
            
            if not summaries:
                print("No summaries found in database")
                return 0
            
            print(f"Found {len(summaries)} summaries in database")
            print(f"Writing to: {output_file}")
            
            # Write all summaries to file with proper UTF-8 encoding
            with open(output_file, "w", encoding="utf-8", errors="replace") as f:
                for i, summary in enumerate(summaries, 1):
                    summary_text = summary['summary_text']
                    if not summary_text:
                        continue
                    
                    f.write(f"{'='*80}\n")
                    f.write(f"Summary {i}/{len(summaries)}")
                    f.write(f" (ID: {summary['id']}, Created: {summary['created_at']})\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(summary_text)
                    f.write("\n\n")
            
            file_size = output_file.stat().st_size
            print(f"✅ Successfully wrote {len(summaries)} summaries to {output_file}")
            print(f"   File size: {file_size:,} bytes")
            return len(summaries)
            
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract all conversations and summaries from database to separate files"
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=Path(DEFAULT_CONVERSATIONS_FILE),
        help=f"Output file for original conversations (default: {DEFAULT_CONVERSATIONS_FILE})"
    )
    parser.add_argument(
        "--summaries",
        type=Path,
        default=Path(DEFAULT_SUMMARIES_FILE),
        help=f"Output file for processed summaries (default: {DEFAULT_SUMMARIES_FILE})"
    )
    
    args = parser.parse_args()
    conversations_file = args.conversations.resolve()
    summaries_file = args.summaries.resolve()
    
    try:
        print("Extracting conversations and summaries from database...\n")
        
        # Extract conversations
        conv_count = extract_all_conversations(conversations_file)
        
        print()  # Blank line between outputs
        
        # Extract summaries
        summary_count = extract_all_summaries(summaries_file)
        
        print(f"\n{'='*50}")
        print(f"Extraction complete:")
        print(f"  Conversations: {conv_count} written to {conversations_file}")
        print(f"  Summaries: {summary_count} written to {summaries_file}")
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}", file=sys.stderr)
        print("\nPlease ensure DATABASE_URL is set in your .env file", file=sys.stderr)
        sys.exit(1)
    except psycopg2.OperationalError as e:
        print(f"❌ Database connection error: {e}", file=sys.stderr)
        print("\nPlease ensure PostgreSQL is running:", file=sys.stderr)
        print("  docker-compose up -d postgres", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

