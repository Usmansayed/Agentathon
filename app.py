import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from main import run_crew


def _parse_ranked_pairs(answer: str, value_suffix: str = "") -> pd.DataFrame:
    rows = []
    for chunk in answer.split(";"):
        item = chunk.strip()
        if not item or "=" not in item:
            continue
        label, raw_val = item.split("=", 1)
        cleaned = raw_val.strip().replace("%", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        rows.append({"label": label.strip(), "value": value, "display": f"{value:.2f}{value_suffix}"})
    return pd.DataFrame(rows)


def _parse_q3_metrics(answer: str) -> dict[str, int]:
    metrics: dict[str, int] = {}
    for key, val in re.findall(r"([a-z_]+)\s*=\s*(-?\d+)", answer):
        metrics[key] = int(val)
    return metrics


def _split_output(formatted_output: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in formatted_output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        output[key.strip()] = value.strip()
    return output


st.set_page_config(page_title="RetailIQ - Agent Dashboard", layout="wide")
load_dotenv()

st.title("RetailIQ Autonomous Squad")
st.caption("Live demo UI for multi-agent execution and business insights")

st.sidebar.header("Configuration")
data_file = st.sidebar.text_input("Data Path", "train_data.csv")
output_file = st.sidebar.text_input("Output File", "team-name.txt")
fallback_only = st.sidebar.checkbox("Fallback-only (no model calls)", value=False)

run_clicked = st.sidebar.button("Run Autonomous Agent", type="primary")

if run_clicked:
    with st.status("Agents are thinking...", expanded=True) as status:
        st.write("Initializing squad and validating dataset...")
        try:
            result = run_crew(data_path=data_file, output_file=output_file, fallback_only=fallback_only)
        except Exception as exc:  # noqa: BLE001 - UI-level error surface
            status.update(label="Run failed", state="error")
            st.error(f"Execution failed: {type(exc).__name__}: {exc}")
        else:
            status.update(label="Analysis complete", state="complete")
            st.success(f"Submission file generated: `{result['output_file']}`")
            if result["used_fallback_models"]:
                st.warning("Primary model set failed once. Run succeeded using fallback model assignments.")

            output_map = _split_output(str(result["formatted_output"]))

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Final Output")
                st.code(str(result["formatted_output"]), language="text")
            with col2:
                st.subheader("Executive Summary (Q5)")
                st.write(output_map.get("Q5", "No Q5 output found."))

            st.subheader("Ranked Insights")
            q1_df = _parse_ranked_pairs(output_map.get("Q1", ""))
            q2_df = _parse_ranked_pairs(output_map.get("Q2", ""))
            q4_df = _parse_ranked_pairs(output_map.get("Q4", ""), value_suffix="%")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Q1: Revenue by Category**")
                st.bar_chart(q1_df.set_index("label")["value"] if not q1_df.empty else pd.Series(dtype=float))
            with c2:
                st.markdown("**Q2: Avg Delivery Days by Region**")
                st.bar_chart(q2_df.set_index("label")["value"] if not q2_df.empty else pd.Series(dtype=float))
            with c3:
                st.markdown("**Q4: Return Rate by Payment Method**")
                st.bar_chart(q4_df.set_index("label")["value"] if not q4_df.empty else pd.Series(dtype=float))

            st.subheader("Data Quality Audit (Q3)")
            q3_metrics = _parse_q3_metrics(output_map.get("Q3", ""))
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Duplicates", q3_metrics.get("duplicates", 0))
            m2.metric("Qty Outliers", q3_metrics.get("quantity_outliers", 0))
            m3.metric("Price Format Errors", q3_metrics.get("price_format_errors", 0))
            m4.metric("Invalid Discounts", q3_metrics.get("invalid_discounts", 0))
            m5.metric("Total Nulls", q3_metrics.get("total_nulls", 0))

            if Path(data_file).exists():
                df = pd.read_csv(data_file)
                d1, d2, d3 = st.columns(3)
                d1.metric("Total Records", len(df))
                d2.metric("Columns", len(df.columns))
                d3.metric("Null Cells", int(df.isna().sum().sum()))
else:
    st.info("Set config in the sidebar and click `Run Autonomous Agent`.")
