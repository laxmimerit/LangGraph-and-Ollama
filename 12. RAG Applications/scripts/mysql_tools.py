# # SQL Agent with LangGraph
# https://github.com/fracpete/employees-db-sqlite

from typing_extensions import TypedDict, Annotated
from typing import List
import os
import re 
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain_community.utilities import SQLDatabase

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

# LLM_MODEL = "qwen3"
LLM_MODEL = "gpt-oss"
BASE_URL = "http://localhost:11434"

llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL, reasoning=True)

response = llm.invoke("Hello, how are you?")
response.pretty_print()

response.additional_kwargs['reasoning_content']

# =============================================================================
# Database Setup
# =============================================================================

def get_db_connection():
    db = SQLDatabase.from_uri('sqlite:///db/employees_db-full-1.0.6.db')
    return db


db = get_db_connection()
tables = db.get_usable_table_names()
SCHEMA = db.get_table_info()

# print(SCHEMA)

# ### SQL Tools

# =============================================================================
# SQL Tools
# =============================================================================

@tool
def get_database_schema(table_name: str = None):
    """Get database schema information for SQL query generation.
    Use this first to understand table structure before creating queries."""
    
    db = get_db_connection()
    
    if table_name:
        tables = db.get_usable_table_names()
        if table_name.lower() in [t.lower() for t in tables]:
            result = db.get_table_info([table_name])
            return result
        
        else:
            return f"Error: Table '{table_name}' not found. Available tables: '{', '.join(tables)}'"
    else:
        return SCHEMA


@tool
def generate_sql_query(question: str, schema_info: str=None):
    """Generate a SQL SELECT query from a natural language question using database schema.
        Always use this after getting schema information."""
    
    schema_to_use = schema_info if schema_info else SCHEMA

    prompt = f"""Based on this database schema:
                {schema_to_use}

                Generate a SQL query to answer this question: {question}

                Rules:
                - Use only SELECT statements
                - Include only existing columns and tables
                - Add appropriate WHERE, GROUP BY, ORDER BY clauses as needed
                - Limit results to 10 rows unless specified otherwise
                - Use proper SQL syntax for SQLite

                Return only the SQL query, nothing else."""
    
    response = llm.invoke(prompt)
    sql_query = response.content.strip()
    print(f"[TOOL] Generated SQL Query: {sql_query[:30]}...")
    return sql_query



@tool
def validate_sql_query(query: str):
    """Validate SQL query for safety and syntax before execution.
        Returns 'Valid: <query>' if safe or 'Error: <message>' if unsafe."""
    
    clean_query = query.strip()

    # remove sql code block
    clean_query = re.sub(r'```sql\s*', '', clean_query, flags=re.IGNORECASE)
    clean_query = re.sub(r'```\s*', '', clean_query, flags=re.IGNORECASE)

    clean_query = clean_query.strip().rstrip(';')

    if not clean_query.lower().startswith('select'):
        return "Error: only 'select' statements are allowed."
    
    # Check 2: Block dangerous SQL keywords
    dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'ALTER', 'DROP', 'CREATE', 'TRUNCATE']
    query_upper = clean_query

    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return f"Error: {keyword} operations are not allowed."
        
    print("[TOOL] Your sql query is validated. Passed!")
    return clean_query


@tool
def execute_sql_query(sql_query: str):
    """Execute a validated SQL query and return results.
    Only use this after validating the query for safety."""

    db = get_db_connection()

    query = validate_sql_query.invoke(sql_query)
    if query.startswith('Error:'):
        return f"Query '{sql_query}' validation failed with Error: {query}"
    
    result = db.run(query)

    if result:
        return  f"Query Results: {result}"
    
    else:
        return f"Query Executed Sucessfully but No Result was Found!"
    
    
@tool
def fix_sql_error(original_query: str, error_message: str, question):
    """Fix a failed SQL query by analyzing the error and generating a corrected version.
        Use this when validation or execution fails."""
    
    fix_prompt = f"""The following SQL query failed:
                    Query: {original_query}
                    Error: {error_message}
                    Original Question: {question}

                    Database Schema:
                    {SCHEMA}

                    Analyze the error and provide a corrected SQL query that:
                    1. Fixes the specific error mentioned
                    2. Still answers the original question
                    3. Uses only valid table and column names from the schema
                    4. Follows SQLite syntax rules

                    Return only the corrected SQL query, nothing else."""
    
    response = llm.invoke(fix_prompt)
    query = response.content.strip()

    print(f"[TOOL] Generated fixed SQL Query.")

    return query

ALL_SQL_TOOLS = [
    get_database_schema,
    generate_sql_query,
    execute_sql_query,
    fix_sql_error
]















