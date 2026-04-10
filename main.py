from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import duckdb
from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

from tools.duckdb_tool import DuckDBTool

DATA_PATH = "train_data.csv"
OUTPUT_FILE = "team-name.txt"


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def build_llm(model_name: str, api_key_env: str, base_url_env: str) -> LLM:
    api_key = _require_env(api_key_env)
    base_url = _require_env(base_url_env)
    return LLM(model=model_name, api_key=api_key, base_url=base_url, temperature=0)


def create_crew(data_path: str) -> Crew:
    duckdb_tool = DuckDBTool(data_path=data_path)

    manager = Agent(
        role="Manager",
        goal="Coordinate specialists to solve all RetailIQ questions accurately and in strict output format.",
        backstory=(
            "You are the orchestration lead. You delegate to specialists, verify consistency, "
            "and ensure the final output is submission-ready."
        ),
        llm=build_llm(
            model_name="deepseek/deepseek-v3.2-speciale",
            api_key_env="CKAI_API_KEY_MANAGER",
            base_url_env="CKAI_BASE_URL_MANAGER",
        ),
        allow_delegation=True,
        verbose=True,
    )

    data_engineer = Agent(
        role="Data Engineer",
        goal="Use DuckDB SQL to compute reliable metrics from the dataset.",
        backstory="You are an expert in SQL analytics, data typing, and robust data quality checks.",
        llm=build_llm(
            model_name="google/gemini-3-flash-preview",
            api_key_env="CKAI_API_KEY_DATA_ENGINEER",
            base_url_env="CKAI_BASE_URL_DATA_ENGINEER",
        ),
        tools=[duckdb_tool],
        allow_delegation=False,
        verbose=True,
    )

    business_analyst = Agent(
        role="Business Analyst",
        goal="Interpret analytical results and rank findings with business clarity.",
        backstory="You translate raw metrics into clear rankings and business signals.",
        llm=build_llm(
            model_name="x-ai/grok-4.1-fast",
            api_key_env="CKAI_API_KEY_BUSINESS_ANALYST",
            base_url_env="CKAI_BASE_URL_BUSINESS_ANALYST",
        ),
        tools=[duckdb_tool],
        allow_delegation=False,
        verbose=True,
    )

    executive_reporter = Agent(
        role="Executive Reporter",
        goal="Create an exactly formatted final submission and concise executive summary.",
        backstory="You write concise executive outputs under strict formatting constraints.",
        llm=build_llm(
            model_name="openai/gpt-5-mini",
            api_key_env="CKAI_API_KEY_EXECUTIVE_REPORTER",
            base_url_env="CKAI_BASE_URL_EXECUTIVE_REPORTER",
        ),
        allow_delegation=False,
        verbose=True,
    )

    q1_task = Task(
        description=(
            "Answer Q1 using DuckDB SQL on table `orders`: compute total revenue per category "
            "with revenue formula quantity * unit_price * (1 - discount/100). Rank highest to lowest. "
            "Return compact ranked labeled values."
        ),
        expected_output="Ranked category revenue list with category and total revenue values.",
        agent=data_engineer,
    )

    q2_task = Task(
        description=(
            "Answer Q2 using DuckDB SQL on table `orders`: compute average delivery time per region, "
            "rank highest to lowest, and return labeled ranked values."
        ),
        expected_output="Ranked region average delivery-time list.",
        agent=data_engineer,
        context=[q1_task],
    )

    q3_task = Task(
        description=(
            "Answer Q3 using DuckDB SQL on table `orders`. Return exactly five counts: "
            "duplicate order_id rows, quantity outliers (>1000), price format errors (non-numeric), "
            "invalid discounts (<0 or >100), and total null cells in raw data."
        ),
        expected_output=(
            "A single line listing five counts with labels: duplicates, quantity_outliers, "
            "price_format_errors, invalid_discounts, total_nulls."
        ),
        agent=data_engineer,
    )

    q4_task = Task(
        description=(
            "Answer Q4 using DuckDB SQL on table `orders`: compute return rate (%) per payment method, "
            "rank highest to lowest, and return labeled ranked values."
        ),
        expected_output="Ranked payment-method return-rate list in percentages.",
        agent=business_analyst,
        context=[q1_task, q2_task, q3_task],
    )

    q5_task = Task(
        description=(
            "Answer Q5 in exactly 3 sentences based on Q1-Q4 outputs. Keep it executive-level, "
            "fact-based, and concise."
        ),
        expected_output="Exactly 3 sentences executive summary.",
        agent=executive_reporter,
        context=[q1_task, q2_task, q3_task, q4_task],
    )

    final_task = Task(
        description=(
            "Create final response in exact format only:\n"
            "Q1: [Answer]\n"
            "Q2: [Answer]\n"
            "Q3: [5 specific counts]\n"
            "Q4: [Answer]\n"
            "Q5: [Exactly 3 sentences]\n"
            "No markdown, no extra text."
        ),
        expected_output="Final 5-line plain text content that strictly follows required labels and ordering.",
        agent=manager,
        context=[q1_task, q2_task, q3_task, q4_task, q5_task],
    )

    return Crew(
        agents=[data_engineer, business_analyst, executive_reporter],
        tasks=[q1_task, q2_task, q3_task, q4_task, q5_task, final_task],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=True,
    )


def _extract_five_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    wanted = []
    for key in ("Q1:", "Q2:", "Q3:", "Q4:", "Q5:"):
        match = next((ln for ln in lines if ln.startswith(key)), "")
        wanted.append(match)
    return "\n".join(wanted).strip()


def _enforce_q5_three_sentences(formatted_text: str) -> str:
    lines = formatted_text.splitlines()
    if len(lines) != 5:
        return formatted_text
    q5 = lines[4]
    if not q5.startswith("Q5:"):
        return formatted_text
    body = q5[3:].strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if s]
    if len(sentences) > 3:
        lines[4] = f"Q5: {' '.join(sentences[:3]).strip()}"
    return "\n".join(lines)


def _fallback_answers(data_path: str) -> dict[str, str]:
    conn = duckdb.connect(database=":memory:")
    safe_path = data_path.replace("\\", "/").replace("'", "''")
    conn.execute(
        f"""
        CREATE OR REPLACE VIEW orders AS
        SELECT *
        FROM read_csv_auto('{safe_path}')
        """
    )

    q1_rows = conn.execute(
        """
        SELECT COALESCE(product_category, 'Unknown') AS category,
               SUM(
                   COALESCE(TRY_CAST(quantity AS DOUBLE), 0)
                   * COALESCE(TRY_CAST(unit_price AS DOUBLE), 0)
                   * (1 - COALESCE(TRY_CAST(discount_percent AS DOUBLE), 0) / 100.0)
               ) AS total_revenue
        FROM orders
        GROUP BY 1
        ORDER BY total_revenue DESC
        """
    ).fetchall()
    q1 = "; ".join(f"{cat}={rev:.2f}" for cat, rev in q1_rows)

    q2_rows = conn.execute(
        """
        SELECT COALESCE(customer_region, 'Unknown') AS region,
               AVG(COALESCE(TRY_CAST(delivery_days AS DOUBLE), 0)) AS avg_delivery_time
        FROM orders
        GROUP BY 1
        ORDER BY avg_delivery_time DESC
        """
    ).fetchall()
    q2 = "; ".join(f"{region}={avg_days:.2f}" for region, avg_days in q2_rows)

    duplicates = conn.execute(
        """
        SELECT COALESCE(SUM(cnt - 1), 0)
        FROM (
            SELECT order_id, COUNT(*) AS cnt
            FROM orders
            GROUP BY order_id
            HAVING COUNT(*) > 1
        ) t
        """
    ).fetchone()[0]
    quantity_outliers = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE COALESCE(TRY_CAST(quantity AS DOUBLE), 0) > 1000"
    ).fetchone()[0]
    price_format_errors = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE unit_price IS NOT NULL AND TRY_CAST(unit_price AS DOUBLE) IS NULL"
    ).fetchone()[0]
    invalid_discounts = conn.execute(
        """
        SELECT COUNT(*)
        FROM orders
        WHERE discount_percent IS NOT NULL
          AND (TRY_CAST(discount_percent AS DOUBLE) < 0 OR TRY_CAST(discount_percent AS DOUBLE) > 100)
        """
    ).fetchone()[0]
    total_nulls = conn.execute(
        """
        SELECT SUM(
            (order_id IS NULL)::INT + (date IS NULL)::INT + (product_category IS NULL)::INT +
            (product_name IS NULL)::INT + (quantity IS NULL)::INT + (unit_price IS NULL)::INT +
            (discount_percent IS NULL)::INT + (customer_region IS NULL)::INT +
            (payment_method IS NULL)::INT + (delivery_days IS NULL)::INT + (return_status IS NULL)::INT
        )
        FROM orders
        """
    ).fetchone()[0]
    q3 = (
        f"duplicates={duplicates}, quantity_outliers={quantity_outliers}, "
        f"price_format_errors={price_format_errors}, invalid_discounts={invalid_discounts}, total_nulls={total_nulls}"
    )

    q4_rows = conn.execute(
        """
        SELECT COALESCE(payment_method, 'Unknown') AS payment_method,
               100.0 * AVG(
                   CASE
                       WHEN LOWER(COALESCE(return_status, '')) IN ('returned', 'yes', 'true', '1') THEN 1.0
                       ELSE 0.0
                   END
               ) AS return_rate_pct
        FROM orders
        GROUP BY 1
        ORDER BY return_rate_pct DESC
        """
    ).fetchall()
    q4 = "; ".join(f"{method}={rate:.2f}%" for method, rate in q4_rows)

    top_cat, top_rev = q1_rows[0]
    top_region, top_days = q2_rows[0]
    top_payment, top_rate = q4_rows[0]
    q5 = (
        f"Revenue is led by {top_cat} at {top_rev:.2f}, with clear concentration in top categories. "
        f"{top_region} has the longest average delivery time at {top_days:.2f} days, signaling fulfillment pressure in that region. "
        f"Return behavior is highest for {top_payment} at {top_rate:.2f}%, and the detected data quality issues should be cleaned before executive reporting."
    )

    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5}


def _build_strict_output(text_result: str, fallback: dict[str, str]) -> str:
    extracted = _extract_five_lines(text_result)
    candidate = _enforce_q5_three_sentences(extracted).strip()
    lines = [line.strip() for line in candidate.splitlines() if line.strip()]
    by_key = {line.split(":", 1)[0]: line for line in lines if ":" in line}

    final_lines = []
    for key in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        line = by_key.get(key, "").strip()
        if not line or line == f"{key}:":
            line = f"{key}: {fallback[key]}"
        line = re.sub(r"\s+", " ", line)
        final_lines.append(line)

    # Guarantee exactly 3 sentences for Q5.
    q5_body = final_lines[4].split(":", 1)[1].strip()
    q5_sentences = [s for s in re.split(r"(?<=[.!?])\s+", q5_body) if s]
    if len(q5_sentences) < 3:
        final_lines[4] = f"Q5: {fallback['Q5']}"
    else:
        final_lines[4] = f"Q5: {' '.join(q5_sentences[:3])}"

    return "\n".join(final_lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    load_dotenv()

    data_path = Path(DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(f"DATA_PATH does not exist: {data_path}")

    crew = create_crew(str(data_path))
    result = crew.kickoff()
    text_result = str(result)
    fallback = _fallback_answers(str(data_path.resolve()))
    formatted = _build_strict_output(text_result, fallback)

    output_path = Path(OUTPUT_FILE)
    output_path.write_text(formatted + "\n", encoding="utf-8")
    print(f"Wrote submission file: {output_path.resolve()}")
    print("-----")
    print(formatted)


if __name__ == "__main__":
    main()
