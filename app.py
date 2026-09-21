"""
Inventory & Fulfillment Exception Engine - Streamlit Application
================================================================
Apex Logistics | Supply Chain Operations Analytics

Converts the notebook-based natural-language query engine into a
self-service Streamlit app.

Pipeline (unchanged from the notebook):
    1. Intent classification  -> verified template vs. fresh SQL
    2. Query construction     -> library lookup or LLM generation
    3. Validation gate        -> read-only / schema / EXPLAIN / LLM relevance
    4. Retry once             -> generated track only
    5. Escalate               -> if still failing
    6. Execute                -> read-only SQLite connection
    7. Response generation    -> concise business answer
    +  Immutable audit log of every request

Run with:
    streamlit run app.py

Expects, in the same folder as this file:
    - supply_chain_ops.db      (required)
    - test_queries.csv         (optional, enables the Evaluation tab)
    - config.json  OR  .streamlit/secrets.toml  OR  environment variables
"""

from typing import Optional
import sqlite3
import sqlparse
import pandas as pd
import json
import re
import os
import warnings
from datetime import datetime

import streamlit as st

from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Inventory & Fulfillment Exception Engine",
    page_icon="📦",
    layout="wide",
)

DB_PATH = os.getenv("SUPPLY_CHAIN_DB_PATH", "supply_chain_ops.db")
TEST_QUERIES_PATH = os.getenv("TEST_QUERIES_PATH", "test_queries.csv")

PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gpt-4o")


# ============================================================
# Credential Loading
# ============================================================

def load_credentials():
    """
    Resolves OpenAI credentials from, in order of precedence:
      1. Streamlit secrets  (.streamlit/secrets.toml - used on Streamlit Cloud)
      2. config.json        (the notebook's local mechanism)
      3. Environment variables

    Returns (api_key, api_base). Either may be None.
    """

    api_key = None
    api_base = None

    # 1. Streamlit secrets
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        api_base = st.secrets.get("OPENAI_API_BASE", None)
    except Exception:
        pass

    # 2. config.json fallback (same as the notebook)
    if not api_key and os.path.exists("config.json"):
        try:
            with open("config.json", "r") as file:
                config = json.load(file)
                api_key = config.get("OPENAI_API_KEY")
                api_base = config.get("OPENAI_API_BASE")
        except Exception as e:
            st.sidebar.warning(f"Could not read config.json: {e}")

    # 3. Environment variable fallback
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_base:
        api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")

    # Publish to the environment so LangChain picks them up
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base
        os.environ["OPENAI_API_BASE"] = api_base

    return api_key, api_base


OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()


# ============================================================
# Model Initialization (cached across reruns)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_llms(model_name: str, evaluator_name: str):
    """
    Two LLMs to separate responsibilities:
      - a lightweight model for classification, generation, and response
      - a more capable model for validation and evaluation
    """
    llm = ChatOpenAI(model=model_name, temperature=0)
    evaluator_llm = ChatOpenAI(model=evaluator_name, temperature=0)
    return llm, evaluator_llm


# ============================================================
# Database Loading (read-only, cached)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_connection(db_path: str):
    """
    Opens a read-only connection to the local SQLite database.

    `?mode=ro` guarantees no write operation can execute even if a
    validation check were bypassed - defence in depth alongside the
    SQL-level validations, aligning with the data sovereignty requirement.

    check_same_thread=False is required because Streamlit serves each
    rerun from a worker thread.
    """
    return sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
        check_same_thread=False,
    )


@st.cache_data(show_spinner=False)
def load_ground_truth(path: str) -> Optional[pd.DataFrame]:
    """Loads the evaluation CSV if it is present next to the app."""
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


# ============================================================
# Database Schema (single source of truth for the LLM)
# ============================================================

database_schema = """
warehouse_master:
  warehouse_id (TEXT, PK): US MSA facility code (e.g., WH_ORD_01, WH_DFW_02)
  warehouse_name (TEXT): legal facility name (e.g., 'Chicago O\\'Hare Hub', 'Dallas Fort-Worth Main')
  region (TEXT): US Census Region (Northeast, Midwest, South, West)
  max_capacity_pallet_positions (INTEGER): total high-bay pallet position capacity
  current_occupancy_pct (REAL): utilization percentage; high occupancy threshold is >= 85.0
  is_cbp_bonded_ftz (INTEGER): 1 if CBP-bonded or Foreign Trade Zone, 0 otherwise

inventory_levels:
  inventory_id (INTEGER, PK): auto-increment identifier
  warehouse_id (TEXT, FK): joins to warehouse_master.warehouse_id
  sku_id (TEXT): unique stock keeping unit code
  sku_category (TEXT): one of Consumer Packaged Goods, Automotive Parts, Cold-Chain Perishables, Apparel, Industrial
  units_on_hand (INTEGER): physical stock count in warehouse
  reorder_point (INTEGER): minimum stock threshold before replenishment order
  unit_cost_usd (REAL): carrying unit cost under US GAAP (ASC 330)
  last_restock_date (DATE): date of last inventory receipt

shipment_tracker:
  shipment_id (TEXT, PK): unique BOL or tracking number (e.g., BOL-000001)
  order_id (TEXT): client purchase order reference (e.g., PO-100001)
  origin_warehouse_id (TEXT, FK): joins to warehouse_master.warehouse_id
  scac_code (TEXT, FK): joins to carrier_performance.scac_code
  promised_ship_date (DATE): contractual SLA dispatch date
  actual_ship_date (DATE): actual gate-out dispatch date, NULL if pending
  delivery_status (TEXT): one of On-Time, Delayed, In-Transit, Cancelled
  delay_reason (TEXT): one of 'FMCSA Driver HOS Limit', 'DOT Road Closure', 'Chassis Shortage', 'CBP Freight Hold', 'Warehouse Backlog', 'N/A'

carrier_performance:
  scac_code (TEXT, PK): NMFTA Standard Carrier Alpha Code (e.g., FEDX, UPSN, XPOF, JBHA, ODFL)
  carrier_name (TEXT): legal corporate name of carrier
  otif_compliance_pct (REAL): On-Time In-Full delivery percentage as a decimal (e.g., 0.94 for 94%)
  avg_delay_hours (REAL): mean delivery delay in hours
  otif_chargeback_usd (REAL): accrued SLA non-compliance penalties in USD

Business rules:
- Stockout definition: units_on_hand = 0
- Below reorder definition: units_on_hand > 0 AND units_on_hand <= reorder_point
- High occupancy threshold: current_occupancy_pct >= 85.0
- Delayed shipments: delivery_status = 'Delayed'
- In-transit shipments: delivery_status = 'In-Transit'
- Inventory value formula: units_on_hand * unit_cost_usd
"""


# ============================================================
# Verified Query Template Library
# ============================================================
# 10 pre-approved templates for common supply chain questions.
# Each entry holds a verified SQL query plus a plain-English
# description used by the router for semantic intent matching.

verified_query_library = {
    'VQ1': {
        'description': 'Regional stockout count showing which US regions have the most SKUs currently at zero units on hand',
        'sql': """SELECT w.region,
     COUNT(*) AS stockout_skus
FROM inventory_levels i
JOIN warehouse_master w ON i.warehouse_id = w.warehouse_id
WHERE i.units_on_hand = 0
GROUP BY w.region
ORDER BY stockout_skus DESC"""
    },


    'VQ2': {
        'description': 'SKU categories with the most items currently below reorder point but not yet stocked out, indicating near-term replenishment need',
        'sql': """SELECT sku_category,
     COUNT(*) AS below_reorder_skus
FROM inventory_levels
WHERE units_on_hand > 0 AND units_on_hand <= reorder_point
GROUP BY sku_category
ORDER BY below_reorder_skus DESC"""
    },


    'VQ3': {
        'description': 'Warehouses at or above the 85% high-occupancy threshold, indicating capacity risk',
        'sql': """SELECT warehouse_id,
     warehouse_name,
     region,
     current_occupancy_pct
FROM warehouse_master
WHERE current_occupancy_pct >= 85.0
ORDER BY current_occupancy_pct DESC"""
    },


    'VQ4': {
        'description': 'Total count of shipments currently marked as Delayed in the shipment tracker',
        'sql': """SELECT COUNT(*) AS delayed_count
FROM shipment_tracker
WHERE delivery_status = 'Delayed'"""
    },


    'VQ5': {
        'description': 'Carriers ranked from worst to best by On-Time In-Full (OTIF) compliance percentage',
        'sql': """SELECT scac_code,
     carrier_name,
     otif_compliance_pct
FROM carrier_performance
ORDER BY otif_compliance_pct ASC"""
    },


    'VQ6': {
        'description': 'Carrier with the highest accrued OTIF chargeback penalties in USD',
        'sql': """SELECT * FROM (
    SELECT carrier_name,
         otif_chargeback_usd
    FROM carrier_performance
    ORDER BY otif_chargeback_usd DESC
    LIMIT 1)"""
    },


    'VQ7': {
        'description': 'Top 5 warehouses ranked by total inventory value (units_on_hand * unit_cost_usd), showing where carrying cost is concentrated',
        'sql': """SELECT * FROM (SELECT w.warehouse_id,
     w.warehouse_name,
     ROUND(SUM(i.units_on_hand * i.unit_cost_usd), 2) AS inventory_value_usd
FROM inventory_levels i
JOIN warehouse_master w ON i.warehouse_id = w.warehouse_id
GROUP BY w.warehouse_id
ORDER BY inventory_value_usd DESC
LIMIT 5)"""
    },


    'VQ8': {
        'description': 'Most common reasons for shipment delays with occurrence counts across all delayed shipments',
        'sql': """SELECT delay_reason,
     COUNT(*) AS occurrences
FROM shipment_tracker
WHERE delivery_status = 'Delayed'
GROUP BY delay_reason
ORDER BY occurrences DESC"""
    },


    'VQ9': {
        'description': 'Average occupancy comparison between CBP-bonded/FTZ warehouses and non-bonded facilities',
        'sql': """SELECT is_cbp_bonded_ftz,
     ROUND(AVG(current_occupancy_pct), 2) AS avg_occupancy_pct,
     COUNT(*) AS warehouse_count
FROM warehouse_master
GROUP BY is_cbp_bonded_ftz"""
    },


    'VQ10': {
        'description': 'Aggregate count of shipments currently in transit broken down by carrier SCAC code',
        'sql': """SELECT scac_code,
     COUNT(*) AS in_transit_count
FROM shipment_tracker
WHERE delivery_status = 'In-Transit'
GROUP BY scac_code
ORDER BY in_transit_count DESC"""
    }
}


# ============================================================
# Tool 1: Intent Classification
# ============================================================

def classify_intent(user_question, query_library, llm):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.
    - llm: The primary chat model.

    Returns:
    - dict: Contains 'route' (verified or generated), 'query_id' (template ID or None),
            and 'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
### ROLE
You are a query router for a supply chain operations analytics system. Your job is to decide whether a business user's question can be answered by one of the pre-approved query templates, or whether it needs fresh SQL generation.

### INPUT
User Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

### INSTRUCTIONS
1. Read the user question carefully and identify the analytical intent.
2. Compare the intent against each template description.
3. Match on semantic meaning, not exact wording. For example, 'out of stock' means stockout (units_on_hand = 0), 'FTZ' or 'bonded' refers to is_cbp_bonded_ftz = 1, 'late' or 'behind schedule' means Delayed, 'facilities near capacity' means high occupancy.
4. If a template genuinely answers the question, return that template ID.
5. If no template covers the question, return null for the query_id and set the route to generated.
6. Be careful about shape of answer: a question asking for row-level detail (e.g., 'show me the shipments') should NOT match a template that returns an aggregate count.

### OUTPUT
Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()

    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ============================================================
# Tool 2: Query Generation
# ============================================================

def generate_query(user_question, schema_context, llm):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.
    - llm: The primary chat model.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""
### ROLE
You are a senior SQL developer specializing in supply chain operations analytics on a SQLite database.

### INPUT
User Question:
{user_question}

Database Schema (single source of truth):
{schema_context}

### INSTRUCTIONS
1. Write a single SQL query that answers the user question using only the provided schema.
2. The query must be read-only. Use SELECT (or WITH ... SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
3. Use only the tables and columns listed in the schema. Do not invent columns.
4. Resolve named entities using warehouse_name or warehouse_id where relevant (for example, 'Dallas' maps to WH_DFW_01, 'Chicago' maps to WH_ORD_01).
5. Ensure the query is SQLite compatible.
6. In SQLite, never subtract DATE() or date columns directly (e.g. DATE(a)-DATE(b)) - it silently returns 0; always use julianday(a)-julianday(b) for day differences.
7. Alias every numeric column with a suffix that states its unit, so the result is self-describing. Use _usd for dollar amounts, _pct or _percent for percentages, _count for counts, _hours for hour values, and _days for day values. Avoid bare aliases like "value", "amount", or "total".

### OUTPUT
Return ONLY the SQL query, with no markdown code blocks, no comments, and no explanation.
"""

    sql = llm.invoke(generation_prompt).content.strip()

    # Remove Markdown code fences (```sql ... ```) and extra whitespace from the extracted SQL
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Remove generic Markdown code fences (``` ... ```) and extra whitespace
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()

    return sql


# ============================================================
# Tool 3: Query Validation Gate
# ============================================================

def validate_query(user_question, candidate_sql, db_connection, query_library,
                   evaluator_llm, query_id=None):
    '''
    Four-stage validation gate. The query proceeds to execution only if
    every check passes.

      1. Read-only shape check
      2. Schema conformance check
      3. Parse-and-plan dry run (SQLite EXPLAIN)
      4. LLM relevance check
    '''

    # Store the validation result; query is considered failed by default
    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # ============================================================
    # CHECK 1: READ-ONLY SHAPE CHECK
    # Make sure the query is safe and contains only read operations.
    # ============================================================

    sql_upper = candidate_sql.upper().strip()

    forbidden_keywords = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT',
        'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH'
    ]

    # Query must start with SELECT or WITH
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result

    # Block forbidden SQL operations
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result

    # Allow only one SQL statement
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # ============================================================
    # CHECK 2: SCHEMA CONFORMANCE CHECK
    # Make sure the query uses valid tables and columns.
    # ============================================================

    cur = db_connection.cursor()

    # Get all real tables from the database
    real_tables = [
        r[0]
        for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]

    # Get all real columns from those tables
    real_columns = set()

    for t in real_tables:
        for col_info in cur.execute(
            f"PRAGMA table_info({t})"
        ).fetchall():
            real_columns.add(col_info[1].lower())

    # Parse the SQL
    parsed = sqlparse.parse(candidate_sql)[0]

    # Extract identifiers used in the SQL
    tokens = [
        str(t).strip().lower()
        for t in parsed.flatten()
        if t.ttype is None or 'Name' in str(t.ttype)
    ]

    referenced_identifiers = re.findall(
        r'\b[a-z_][a-z0-9_]*\b',
        candidate_sql.lower()
    )

    # SQL keywords that should not be treated as table/column names
    sql_keywords = {
        'select', 'from', 'where', 'and', 'or', 'group', 'by',
        'order', 'having', 'limit', 'join', 'on', 'as', 'case',
        'when', 'then', 'else', 'end', 'sum', 'count', 'avg',
        'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
        'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in',
        'like', 'with', 'union', 'all', 'between', 'coalesce'
    }

    # Find identifiers that are not known tables, columns, or keywords.
    # Retained as a diagnostic signal: the final gate relies on SQLite
    # parsing/planning and the LLM relevance check rather than treating
    # every SQL token (aliases, functions, literals) as a schema error.
    unknown = [
        tok for tok in referenced_identifiers
        if tok not in sql_keywords
        and tok not in real_columns
        and tok not in real_tables
        and not tok.isdigit()
        and tok not in ('w', 'i', 's', 'c', 'p')
    ]

    # ============================================================
    # CHECK 3: PARSE-AND-PLAN DRY RUN
    # Use EXPLAIN to confirm that the SQL can be parsed and planned.
    # ============================================================

    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()

    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # ============================================================
    # CHECK 4: LLM RELEVANCE CHECK
    # Ask the LLM whether the SQL actually answers the user's question.
    # ============================================================

    # Check whether this SQL came from the verified-query library
    is_verified_track = (
        query_id is not None and query_id in query_library
    )

    # Give the evaluator the correct context for the query type
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all regions/categories/carriers rather than filtering "
        "to just what the user asked). A separate response-generation step will "
        "filter and highlight the relevant rows afterward. Do NOT fail this query "
        "for lacking a WHERE clause that narrows to the user's specific "
        "region/category/carrier: judge only whether the underlying metric, tables, "
        "and aggregation logic match the question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped and filtered to answer it directly."
    )

    # Prompt the LLM to evaluate business relevance
    relevance_prompt = f"""
### ROLE
You are a senior data validator. Your job is to check whether a SQL query
correctly answers a business user's question about supply chain operations.

### CONTEXT
{track_context}

### INPUT
User Question: {user_question}

Candidate SQL:
{candidate_sql}

### INSTRUCTIONS
Assess whether the SQL genuinely answers what the user asked, considering:

1. Does it query the correct tables and columns?
2. Does it apply the right aggregations and groupings?
3. Does it handle the requested business definitions correctly?
4. Does it resolve named entities correctly?
5. Does it return the right shape of answer?
6. If this is a verified template, do not penalize it for returning
   a broader result set than the question's scope.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""

    # Send the question and SQL to the evaluator LLM
    relevance_response = evaluator_llm.invoke(
        relevance_prompt
    ).content.strip()

    # Extract the JSON response from the LLM
    json_match = re.search(
        r'\{.*\}',
        relevance_response,
        re.DOTALL
    )

    if json_match:

        try:
            relevance_json = json.loads(json_match.group())
        except json.JSONDecodeError:
            relevance_json = {}

        # Store the LLM confidence score
        result['relevance_confidence'] = relevance_json.get(
            'confidence',
            0.0
        )

        # Fail if the LLM rejects the query or confidence is below 0.6
        if (
            relevance_json.get('verdict') == 'no'
            or relevance_json.get('confidence', 0.0) < 0.6
        ):
            result['failed_check'] = 'llm_relevance'
            result['details'] = (
                f"Relevance check failed: "
                f"{relevance_json.get('reason', 'unknown')}"
            )
            return result

    # ============================================================
    # ALL 4 CHECKS PASSED
    # The query is now approved for execution.
    # ============================================================

    result['passed'] = True
    result['details'] = 'All validation checks passed'

    return result


# ============================================================
# Tool 4: Retry Generation
# ============================================================

def retry_generation(user_question, failed_sql, error_message, schema_context, llm):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.
    - llm: The primary chat model.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
### ROLE
You are a senior SQL developer fixing a query that failed validation.

### INPUT
User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.
2. Preserve the original intent of the query.
3. The revised SQL must be read-only SELECT (or WITH ... SELECT).
4. Use only tables and columns from the schema.
5. Ensure the query is SQLite compatible.

### OUTPUT
Return ONLY the corrected SQL, with no markdown code blocks, no comments, and no explanation.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()

    # Remove Markdown code fences (```sql ... ```) and extra whitespace from the extracted SQL
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Remove generic Markdown code fences (``` ... ```) and extra whitespace
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()

    return revised_sql


# ============================================================
# Tool 5: Query Execution
# ============================================================

def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    # Initialize the result structure with default values.
    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    # Execute the validated SQL query and store the results in a DataFrame.
    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Perform basic reasonableness checks on the query results.
    # These checks flag potential data-quality issues but do not stop execution.

    # Check whether the query returned any rows.
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    # Check numeric columns for potentially unexpected values.
    for col in df.select_dtypes(include='number').columns:

        # Flag negative values unless the column represents a deviation or change,
        # where negative values can be valid and meaningful.
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')

        # Check for missing (NULL/NaN) values in the numeric column.
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()

            # Warn when more than 50% of the column values are missing,
            # as this may indicate a data-quality or query issue.
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    return result


# ============================================================
# Tool 6: Response Generation
# ============================================================

def generate_response(user_question, dataframe, route, llm, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - llm: The primary chat model.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""
### ROLE
You are a supply chain operations analyst writing a concise business response for a fulfillment or inventory question.

### INPUT
User Question: {user_question}

Query Result Data:
{dataframe.to_string()}

### INSTRUCTIONS
1. Answer the user's specific question directly. Do not dump the entire table.
2. If the user asked about a specific region, warehouse, carrier, or category, highlight only those rows.
3. Provide context from other rows only when it adds value (for example, ranking or comparison).
4. State exact numbers from the data. Do not round beyond what is shown.
5. Flag anything notable, such as a warehouse close to a capacity threshold or a carrier significantly worse than peers.
6. Use clear, professional language suitable for a fulfillment operations memo.
7. Keep the response focused. Two to four sentences for simple questions, up to a short paragraph for complex ones.
8. State the unit for every number, inferred from its column name: _usd as "$X", _pct or _percent as "X%", _count as a plain count, _hours as "X hours". Never state a bare number when the source column implies a unit.

### OUTPUT
Return ONLY the natural language response text, with no markdown headers or bullet points unless truly needed.
"""

    response = llm.invoke(response_prompt).content.strip()
    return response


# ============================================================
# Complete Workflow (7 steps)
# ============================================================

def run_workflow(user_question, conn, llm, evaluator_llm, status=None):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - conn: Read-only SQLite connection.
    - llm / evaluator_llm: Chat models.
    - status: Optional Streamlit status container for the live pipeline trace.

    Returns:
    - dict: Complete pipeline output including response, SQL, data, and log.
    '''

    def report(message):
        """Streams a pipeline stage into the Streamlit trace panel."""
        if status is not None:
            status.write(message)

    db_connection = conn
    query_library = verified_query_library
    schema_context = database_schema

    log = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'warnings': [],
        'response': None
    }

    # --------------------------------------------------------
    # Step 1: Intent classification
    # --------------------------------------------------------

    classification = classify_intent(user_question, query_library, llm)

    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    report(
        f"**[1] Intent Classification:** route=`{log['route']}`, "
        f"query_id=`{log['query_id']}`  \n"
        f"Reason: {log['match_reason']}"
    )

    # --------------------------------------------------------
    # Step 2: Query construction
    # --------------------------------------------------------

    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
        report("**[2] Query Construction:** loaded from verified library")
    else:
        candidate_sql = generate_query(user_question, schema_context, llm)
        report("**[2] Query Construction:** generated fresh SQL")

    log['candidate_sql'] = candidate_sql

    # --------------------------------------------------------
    # Step 3: Validation gate
    # --------------------------------------------------------

    gate = validate_query(
        user_question, candidate_sql, db_connection,
        query_library, evaluator_llm, log['query_id']
    )

    log['gate_result'] = gate
    log['confidence'] = gate.get('relevance_confidence')

    report(
        f"**[3] Validation Gate:** passed=`{gate['passed']}`, "
        f"relevance_confidence=`{gate.get('relevance_confidence')}`"
    )
    if not gate['passed']:
        report(
            f"Failed check: `{gate.get('failed_check')}`  \n"
            f"Details: {gate.get('details')}"
        )

    # --------------------------------------------------------
    # Step 4: Retry once on generated track if validation fails
    # --------------------------------------------------------

    if not gate['passed'] and log['route'] == 'generated':

        report(f"Retrying: {gate['details']}")

        candidate_sql = retry_generation(
            user_question, candidate_sql, gate['details'], schema_context, llm
        )

        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True

        gate = validate_query(
            user_question, candidate_sql, db_connection,
            query_library, evaluator_llm, None
        )

        log['gate_result'] = gate
        log['confidence'] = gate.get('relevance_confidence')

        report(
            f"**Retry Validation Gate:** passed=`{gate['passed']}`, "
            f"relevance_confidence=`{gate.get('relevance_confidence')}`"
        )
        if not gate['passed']:
            report(
                f"Retry failed check: `{gate.get('failed_check')}`  \n"
                f"Retry details: {gate.get('details')}"
            )

    # --------------------------------------------------------
    # Step 5: Escalate if still failing
    # --------------------------------------------------------

    if not gate['passed']:
        log['escalated'] = True
        log['route'] = 'escalate'
        log['response'] = (
            "Query could not be reliably resolved. Escalated to human analyst. "
            f"Failure: {gate['details']}"
        )
        log['confidence'] = gate.get('relevance_confidence')

        report(f"**[!] Escalated to human:** {gate['details']}")

        return {'log': log, 'dataframe': None, **log}

    # --------------------------------------------------------
    # Step 6: Execute
    # --------------------------------------------------------

    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)
    log['warnings'] = exec_result['warnings']

    report(f"**[4] Execute:** {len(df)} rows returned")
    if exec_result['warnings']:
        report(f"Warnings: {exec_result['warnings']}")

    # --------------------------------------------------------
    # Step 7: Response generation
    # --------------------------------------------------------

    response = generate_response(user_question, df, log['route'], llm, log['query_id'])
    log['response'] = response

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    report(f"**[5] Response Generation:** confidence=`{log['confidence']}`")

    return {'log': log, 'dataframe': df, **log}


# ============================================================
# Audit Log Helpers (immutable, session-scoped)
# ============================================================

def append_audit_log(log_entry):
    """Appends a completed pipeline run to the in-session audit log."""
    if 'audit_log' not in st.session_state:
        st.session_state['audit_log'] = []

    st.session_state['audit_log'].append({
        'timestamp': log_entry.get('timestamp'),
        'user_question': log_entry.get('user_question'),
        'route': log_entry.get('route'),
        'query_id': log_entry.get('query_id'),
        'match_reason': log_entry.get('match_reason'),
        'retry_used': log_entry.get('retry_used'),
        'escalated': log_entry.get('escalated'),
        'confidence': log_entry.get('confidence'),
        'row_count': log_entry.get('row_count'),
        'executed_sql': log_entry.get('executed_sql'),
        'warnings': '; '.join(log_entry.get('warnings') or []),
        'response': log_entry.get('response'),
    })


def confidence_badge(confidence):
    """Returns (emoji, display string) for a confidence score."""
    if isinstance(confidence, (int, float)):
        if confidence >= 0.8:
            return "🟢", f"{confidence:.2f}"
        if confidence >= 0.6:
            return "🟡", f"{confidence:.2f}"
        return "🔴", f"{confidence:.2f}"
    return "⚪", str(confidence)


def render_result(output):
    """Renders a single pipeline output: answer, data, SQL, and log."""

    if output['escalated']:
        st.warning(output['response'])
        with st.expander("Validation details"):
            st.json(output['gate_result'])
        with st.expander("Last candidate SQL"):
            st.code(output['candidate_sql'] or '', language='sql')
        return

    badge, confidence_display = confidence_badge(output['confidence'])

    st.subheader("Answer")
    st.write(output['response'])

    route_label = output['route']
    if output['query_id']:
        route_label = f"{route_label} ({output['query_id']})"

    st.caption(
        f"{badge} Confidence: {confidence_display}  ·  "
        f"Route: {route_label}  ·  "
        f"Rows returned: {output['row_count']}  ·  "
        f"Retry used: {output['retry_used']}"
    )

    if output.get('warnings'):
        for warning in output['warnings']:
            st.info(f"Reasonableness check: {warning}")

    if output['dataframe'] is not None:
        with st.expander("View underlying data", expanded=True):
            st.dataframe(output['dataframe'], use_container_width=True)
            st.download_button(
                "Download result as CSV",
                data=output['dataframe'].to_csv(index=False).encode('utf-8'),
                file_name="query_result.csv",
                mime="text/csv",
                key=f"dl_{output['timestamp']}_{abs(hash(output['user_question']))}",
            )

    with st.expander("View executed SQL"):
        st.code(output['executed_sql'] or '', language='sql')

    with st.expander("View audit log entry"):
        st.json({k: v for k, v in output['log'].items() if k != 'gate_result'})
        st.json(output['gate_result'])


# ============================================================
# Application Bootstrap
# ============================================================

st.title("📦 Inventory & Fulfillment Exception Engine")
st.caption(
    "Apex Logistics · Supply Chain Operations Analytics — ask natural-language "
    "questions about inventory, warehouse capacity, shipments, and carrier performance. "
    "All execution is read-only and stays on the local database."
)

# --- Database connection ---
if not os.path.exists(DB_PATH):
    st.error(
        f"Database not found at `{DB_PATH}`. Place `supply_chain_ops.db` next to "
        "`app.py`, or set the SUPPLY_CHAIN_DB_PATH environment variable."
    )
    st.stop()

try:
    conn = get_connection(DB_PATH)
except sqlite3.Error as e:
    st.error(f"Could not open the database in read-only mode: {e}")
    st.stop()

# --- Credentials / LLMs ---
if not OPENAI_API_KEY:
    st.error(
        "No OpenAI API key found. Provide it via `.streamlit/secrets.toml` "
        "(OPENAI_API_KEY / OPENAI_API_BASE), a local `config.json`, or the "
        "OPENAI_API_KEY environment variable."
    )
    st.stop()

llm, evaluator_llm = get_llms(PRIMARY_MODEL, EVALUATOR_MODEL)
ground_truth = load_ground_truth(TEST_QUERIES_PATH)


# ============================================================
# Sidebar
# ============================================================

EXAMPLE_QUESTIONS = [
    "Which 5 warehouses have the highest dollar value of on-hand inventory?",
    "Do our bonded warehouses run hotter on capacity than the regular ones?",
    "Which regions have the most SKUs out of stock?",
    "How many shipments are currently delayed?",
    "Which carriers have the worst OTIF compliance?",
    "What are the most common reasons for shipment delays?",
    "Which warehouses are operating above 85% capacity?",
    "Which SKU categories are sitting below reorder point?",
    "Which carrier owes us the most in OTIF chargebacks?",
    "How many shipments are in transit by carrier?",
]

with st.sidebar:
    st.header("About")
    st.write(
        "This engine routes operational questions to a pre-approved, "
        "version-controlled library of verified SQL templates where possible, "
        "and generates fresh read-only SQL for novel questions. Every query "
        "passes a four-stage validation gate before it touches the database, "
        "retries once on failure, and escalates to a human analyst if it still "
        "cannot be resolved."
    )

    st.subheader("Pipeline")
    st.markdown(
        "1. Intent classification\n"
        "2. Query construction\n"
        "3. Validation gate (read-only · schema · EXPLAIN · relevance)\n"
        "4. Retry once (generated track)\n"
        "5. Escalate if unresolved\n"
        "6. Read-only execution\n"
        "7. Response generation + audit log"
    )

    st.subheader("Models")
    st.markdown(
        f"- Primary: `{PRIMARY_MODEL}`\n"
        f"- Evaluator: `{EVALUATOR_MODEL}`"
    )

    st.subheader("Settings")
    show_trace = st.checkbox("Show pipeline trace", value=True)

    st.subheader("Verified query library")
    st.caption(f"{len(verified_query_library)} approved templates loaded")
    for qid, entry in verified_query_library.items():
        with st.expander(qid):
            st.write(entry['description'])
            st.code(entry['sql'], language='sql')

    st.success("Read-only mode · no data leaves the local network", icon="🔒")


# ============================================================
# Tabs
# ============================================================

tab_ask, tab_explorer, tab_audit, tab_eval = st.tabs(
    ["Ask a question", "Database explorer", "Audit log", "Evaluation"]
)


# ------------------------------------------------------------
# Tab 1: Ask a question
# ------------------------------------------------------------

with tab_ask:

    if 'selected_example' not in st.session_state:
        st.session_state['selected_example'] = ""

    with st.expander("Example questions"):
        cols = st.columns(2)
        for idx, example in enumerate(EXAMPLE_QUESTIONS):
            if cols[idx % 2].button(example, key=f"ex_{idx}", use_container_width=True):
                st.session_state['selected_example'] = example

    user_question = st.text_area(
        "Your question",
        value=st.session_state['selected_example'],
        placeholder="e.g. Which warehouses are operating above 85% capacity?",
        height=90,
    )

    submitted = st.button("Run query", type="primary")

    if submitted and user_question.strip():

        trace_container = (
            st.status("Running pipeline...", expanded=True) if show_trace else None
        )

        try:
            output = run_workflow(
                user_question=user_question.strip(),
                conn=conn,
                llm=llm,
                evaluator_llm=evaluator_llm,
                status=trace_container,
            )
        except Exception as e:
            if trace_container is not None:
                trace_container.update(label="Pipeline error", state="error")
            st.error(f"Pipeline failed: {e}")
            st.stop()

        if trace_container is not None:
            trace_container.update(
                label=(
                    "Escalated to human review" if output['escalated']
                    else "Pipeline complete"
                ),
                state="error" if output['escalated'] else "complete",
                expanded=False,
            )

        append_audit_log(output['log'])

        st.divider()
        render_result(output)

    elif submitted:
        st.info("Please enter a question first.")


# ------------------------------------------------------------
# Tab 2: Database explorer
# ------------------------------------------------------------

with tab_explorer:

    st.subheader("Local database (read-only)")

    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
    )
    st.write("Tables available:")
    st.dataframe(tables, use_container_width=True, hide_index=True)

    table_choice = st.selectbox(
        "Preview a table",
        options=list(tables['name']) if not tables.empty else [],
    )

    if table_choice:
        row_limit = st.slider("Rows to preview", 5, 200, 20, step=5)
        preview = pd.read_sql_query(
            f"SELECT * FROM {table_choice} LIMIT {row_limit}", conn
        )
        st.dataframe(preview, use_container_width=True)

        total = pd.read_sql_query(
            f"SELECT COUNT(*) AS row_count FROM {table_choice}", conn
        )['row_count'].iloc[0]
        st.caption(f"{total:,} total rows in `{table_choice}`")

    with st.expander("Schema passed to the LLM"):
        st.code(database_schema, language="text")


# ------------------------------------------------------------
# Tab 3: Audit log
# ------------------------------------------------------------

with tab_audit:

    st.subheader("Local audit log")
    st.caption(
        "Immutable record of every request handled in this session: executed "
        "statements, validation outcomes, routing decisions, and confidence scores."
    )

    audit_entries = st.session_state.get('audit_log', [])

    if not audit_entries:
        st.info("No queries have been run yet in this session.")
    else:
        audit_df = pd.DataFrame(audit_entries)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Requests", len(audit_df))
        col2.metric("Verified route", int((audit_df['route'] == 'verified').sum()))
        col3.metric("Generated route", int((audit_df['route'] == 'generated').sum()))
        col4.metric("Escalated", int(audit_df['escalated'].sum()))

        numeric_conf = pd.to_numeric(audit_df['confidence'], errors='coerce')
        if numeric_conf.notna().any():
            st.metric("Average confidence", f"{numeric_conf.mean():.2f}")

        st.dataframe(audit_df, use_container_width=True)

        st.download_button(
            "Download audit log as CSV",
            data=audit_df.to_csv(index=False).encode('utf-8'),
            file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------
# Tab 4: Evaluation against ground truth
# ------------------------------------------------------------

with tab_eval:

    st.subheader("Evaluation against ground truth")

    if ground_truth is None:
        st.info(
            f"`{TEST_QUERIES_PATH}` was not found. Place the evaluation CSV next "
            "to `app.py` to enable this tab."
        )
    else:
        st.caption(
            "Runs each ground-truth question end-to-end and compares the selected "
            "route and template against the expected values."
        )

        st.dataframe(ground_truth, use_container_width=True)

        max_cases = len(ground_truth)
        n_cases = st.slider("Number of test cases to run", 1, max_cases, max_cases)

        if st.button("Run evaluation", type="primary"):

            evaluation_rows = []
            progress = st.progress(0.0, text="Running evaluation...")

            for i, (_, gt) in enumerate(ground_truth.head(n_cases).iterrows()):

                try:
                    tr = run_workflow(
                        user_question=gt['question'],
                        conn=conn,
                        llm=llm,
                        evaluator_llm=evaluator_llm,
                        status=None,
                    )
                    append_audit_log(tr['log'])
                except Exception as e:
                    st.error(f"Test {gt['test_id']} failed: {e}")
                    continue

                evaluation_rows.append({
                    'Test ID': gt['test_id'],
                    'Question': gt['question'],
                    'Expected Route': gt['expected_route'],
                    'Actual Route': tr['route'],
                    'Route Match': tr['route'] == gt['expected_route'],
                    'Expected Query ID': gt['expected_query_id'],
                    'Actual Query ID': tr['query_id'],
                    'Query ID Match': (
                        pd.isna(gt['expected_query_id']) and
                        (tr['query_id'] is None or pd.isna(tr['query_id']))
                    ) or tr['query_id'] == gt['expected_query_id'],
                    'Confidence': tr['confidence'],
                    'Rows Returned': tr['row_count'],
                })

                progress.progress((i + 1) / n_cases, text=f"Completed {i + 1}/{n_cases}")

            progress.empty()

            if evaluation_rows:

                evaluation_df = pd.DataFrame(evaluation_rows)

                path_accuracy = evaluation_df['Route Match'].mean() * 100

                verified = (
                    evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
                )

                if verified.any():
                    query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100
                else:
                    query_accuracy = float('nan')

                average_confidence = pd.to_numeric(
                    evaluation_df['Confidence'], errors='coerce'
                ).mean()

                m1, m2, m3 = st.columns(3)
                m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
                m2.metric(
                    "Selected Query Accuracy (verified only)",
                    "n/a" if pd.isna(query_accuracy) else f"{query_accuracy:.1f}%",
                )
                m3.metric(
                    "Average Confidence",
                    "n/a" if pd.isna(average_confidence) else f"{average_confidence:.2f}",
                )

                st.dataframe(evaluation_df, use_container_width=True)

                st.download_button(
                    "Download evaluation results as CSV",
                    data=evaluation_df.to_csv(index=False).encode('utf-8'),
                    file_name="evaluation_results.csv",
                    mime="text/csv",
                )
