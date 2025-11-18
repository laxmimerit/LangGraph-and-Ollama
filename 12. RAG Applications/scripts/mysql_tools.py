# mysql_tools.py

"""
MySQL / SQLite Agent Tools
-----------------------------------
Utility tools for SQL-based AI agents:
- Schema inspection
- Query generation
- Query validation
- Execution
- Error correction
-----------------------------------
Compatible with: LangChain create_agent, ChatOllama, SQLite
"""

import re
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase


# =====================================================================
# GLOBAL VARIABLES
# =====================================================================
# You should initialize `db` and `llm` externally and import them here.
# Example:
# from langchain_ollama import ChatOllama
# db = SQLDatabase.from_uri("sqlite:///db/employees_db-full-1.0.6.db")
# llm = ChatOllama(model="qwen3", base_url="http://localhost:11434")

db = None
llm = None
SCHEMA = None


def init_sql_env(database: SQLDatabase, llm_model, schema: str):
    """Initialize the shared database and model environment."""
    global db, llm, SCHEMA
    db = database
    llm = llm_model
    SCHEMA = schema
    print("Initialized SQL environment")


# =====================================================================
# TOOLS
# =====================================================================

@tool
def get_database_schema(table_name: str = None) -> str:
    """Get database schema information for SQL query generation.
    Use this first to understand table structure before creating queries."""
    if db is None:
        return "Error: Database not initialized. Call init_sql_env() first."

    print(f"Getting schema for: {table_name if table_name else 'all tables'}")

    if table_name:
        try:
            tables = db.get_usable_table_names()
            if table_name.lower() in [t.lower() for t in tables]:
                result = db.get_table_info([table_name])
                print(f"Retrieved schema for table: {table_name}")
                return result
            else:
                return f"Error: Table '{table_name}' not found. Available tables: {', '.join(tables)}"
        except Exception as e:
            return f"Error getting table info: {e}"
    else:
        print("Retrieved full database schema")
        return SCHEMA


@tool
def generate_sql_query(question: str, schema_info: str = None) -> str:
    """Generate a SQL SELECT query from a natural language question using database schema."""
    if llm is None:
        return "Error: LLM not initialized. Call init_sql_env() first."

    print(f"Generating SQL for: {question[:100]}...")
    schema_to_use = schema_info if schema_info else SCHEMA

    prompt = f"""
        Based on this database schema:
        {schema_to_use}

        Generate a SQL query to answer this question: {question}

        Rules:
        - Use only SELECT statements
        - Include only existing columns and tables
        - Add appropriate WHERE, GROUP BY, ORDER BY clauses as needed
        - Limit results to 10 rows unless specified otherwise
        - Use proper SQL syntax for SQLite

        Return only the SQL query, nothing else.
    """

    try:
        response = llm.invoke(prompt)
        query = response.content.strip()
        print("Generated SQL query")
        return query
    except Exception as e:
        return f"Error generating query: {e}"


@tool
def validate_sql_query(query: str) -> str:
    """Validate SQL query for safety and syntax before execution."""
    print(f"Validating SQL: {query[:100]}...")

    clean_query = query.strip()
    clean_query = re.sub(r'```sql\s*', '', clean_query, flags=re.IGNORECASE)
    clean_query = re.sub(r'```\s*', '', clean_query)
    clean_query = clean_query.strip().rstrip(";")

    if not clean_query.lower().startswith("select"):
        return "Error: Only SELECT statements are allowed"

    dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'ALTER', 'DROP', 'CREATE', 'TRUNCATE']
    query_upper = clean_query.upper()

    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return f"Error: {keyword} operations are not allowed"

    print("Query validation passed")
    return f"Valid: {clean_query}"


@tool
def execute_sql_query(query: str) -> str:
    """Execute a validated SQL query and return results."""
    if db is None:
        return "Error: Database not initialized. Call init_sql_env() first."

    print(f"Executing SQL: {query[:100]}...")

    try:
        clean_query = query.strip()
        if clean_query.startswith("Valid: "):
            clean_query = clean_query[7:]
        clean_query = re.sub(r'```sql\s*', '', clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r'```\s*', '', clean_query)
        clean_query = clean_query.strip().rstrip(";")

        result = db.run(clean_query)
        print("Query executed successfully")

        if result:
            return f"Query Results:\n{result}"
        else:
            return "Query executed successfully but returned no results."
    except Exception as e:
        error_msg = f"Execution Error: {str(e)}"
        print(error_msg)
        return error_msg


@tool
def fix_sql_error(original_query: str, error_message: str, question: str) -> str:
    """Fix a failed SQL query by analyzing the error and generating a corrected version."""
    if llm is None:
        return "Error: LLM not initialized. Call init_sql_env() first."

    print(f"Fixing SQL error: {error_message[:100]}...")

    fix_prompt = f"""
        The following SQL query failed:
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

        Return only the corrected SQL query, nothing else.
    """

    try:
        response = llm.invoke(fix_prompt)
        fixed_query = response.content.strip()
        print("Generated fixed SQL query")
        return fixed_query
    except Exception as e:
        return f"Error generating fix: {e}"


# =====================================================================
# TOOL LIST EXPORT
# =====================================================================

ALL_SQL_TOOLS = [
    get_database_schema,
    generate_sql_query,
    validate_sql_query,
    execute_sql_query,
    fix_sql_error
]
 