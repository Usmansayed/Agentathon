# Agentathon - RetailIQ Multi-Agent Solver

This repository contains a production-style Agentathon submission built with a multi-agent architecture.  
It analyzes an e-commerce CSV dataset and writes a strictly formatted `team-name.txt` answer file for Q1-Q5.

## Frameworks and Core Stack

- `CrewAI` for multi-agent orchestration (`Process.hierarchical`)
- `DuckDB` for SQL analytics and deterministic fallback computation
- `python-dotenv` for environment variable loading
- Role-specific LLM routing through CrewAI `LLM(...)` with per-agent base URL and API key

Install dependencies from:

- `requirements.txt` (`crewai[litellm]`, `duckdb`, `python-dotenv`)

## Current Agent Architecture

The implementation is defined in `main.py` and uses 4 agents:

1. **Manager**
   - Role: Orchestrates specialists and composes final output
   - Delegation: Enabled
   - Model slot: `deepseek/deepseek-v3.2-speciale`
2. **Data Engineer**
   - Role: Runs SQL and computes metrics
   - Tooling: `DuckDBTool`
   - Model slot: `google/gemini-3-flash-preview`
3. **Business Analyst**
   - Role: Interprets metrics and ranks insights
   - Tooling: `DuckDBTool`
   - Model slot: `x-ai/grok-4.1-fast`
4. **Executive Reporter**
   - Role: Produces concise executive-ready narrative
   - Model slot: `openai/gpt-5-mini`

## Agent Workflow (Q1-Q5)

The crew executes six tasks in hierarchical mode:

- `Q1` category revenue ranking (using cleaned numeric fields)
- `Q2` regional average delivery-time ranking
- `Q3` five data quality issue counts
- `Q4` payment-method return-rate ranking
- `Q5` exactly 3-sentence executive summary
- `final_task` strict assembly into required 5-line output

Output contract:

```text
Q1: ...
Q2: ...
Q3: ...
Q4: ...
Q5: ...
```

No markdown or extra commentary is allowed in the output file.

## Data and Tooling Layer

`tools/duckdb_tool.py` registers:

- `orders` (raw CSV view)
- `orders_clean` (normalized helper view with:
  - `quantity_num`
  - `unit_price_num`
  - `discount_num`
  - `delivery_days_num`
  - `return_status_norm`)

Agents can run SQL through `duckdb_sql_runner`, and SQL errors are intentionally returned as text so agents can self-correct.

## Reliability Features in Current Implementation

The code is designed to be robust for unseen test CSVs:

- Deterministic fallback computation (`_fallback_answers`) for all Q1-Q5
- Strict post-processing of model output (`_build_strict_output`)
- Missing/malformed sections are auto-replaced with fallback answers
- `Q5` is always forced to deterministic fallback text
- Optional `--fallback-only` mode to skip all model calls

## Environment Variables

Copy `.env.example` to `.env` and fill all variables:

- `CKAI_BASE_URL_MANAGER`
- `CKAI_API_KEY_MANAGER`
- `CKAI_BASE_URL_DATA_ENGINEER`
- `CKAI_API_KEY_DATA_ENGINEER`
- `CKAI_BASE_URL_BUSINESS_ANALYST`
- `CKAI_API_KEY_BUSINESS_ANALYST`
- `CKAI_BASE_URL_EXECUTIVE_REPORTER`
- `CKAI_API_KEY_EXECUTIVE_REPORTER`

## How to Run

From the `Agentathon` directory:

```bash
pip install -r requirements.txt
python main.py --data-path train_data.csv --output-file team-name.txt
```

For deterministic mode (no LLM calls):

```bash
python main.py --data-path train_data.csv --output-file team-name.txt --fallback-only
```

For final evaluation data, only switch the input file path:

```bash
python main.py --data-path test_data.csv --output-file team-name.txt
```

## Important Files

- `main.py` - crew, agents, tasks, strict formatter, fallback pipeline, CLI
- `tools/duckdb_tool.py` - DuckDB query tool + cleaned view creation
- `problem_statement.md` - official challenge rules and question definitions
- `output_format.txt` - mandatory output format
- `team-name.txt` - generated submission file

## Notes for Demo / Explanation

If asked to explain architecture in judging/demo:

- Highlight hierarchical orchestration (`Manager` + specialists)
- Explain SQL-first analytics through DuckDB tooling
- Show strict format enforcement and deterministic fail-safe path
- Emphasize "path-only swap" readiness for unseen `test_data.csv`
