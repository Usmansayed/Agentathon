from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
from crewai.tools import BaseTool
from pydantic import Field, PrivateAttr


class DuckDBTool(BaseTool):
    name: str = "duckdb_sql_runner"
    description: str = (
        "Run SQL against the RetailIQ CSV dataset loaded in DuckDB. "
        "Return query results as plain text. If a query fails, return the error "
        "message so the agent can self-correct."
    )
    data_path: str = Field(..., description="Absolute path to CSV dataset")
    sample_size: int = Field(default=-1, description="DuckDB CSV inference sample size")
    _conn: duckdb.DuckDBPyConnection = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self.data_path = str(Path(self.data_path).resolve())
        self._conn = duckdb.connect(database=":memory:")
        self._register_table()

    def _register_table(self) -> None:
        safe_path = self.data_path.replace("\\", "/").replace("'", "''")
        self._conn.execute(
            f"""
            CREATE OR REPLACE VIEW orders AS
            SELECT *
            FROM read_csv_auto('{safe_path}', SAMPLE_SIZE = {self.sample_size})
            """
        )

    def _format_result(self, rows: list[tuple[Any, ...]], columns: list[str]) -> str:
        if not rows:
            return "OK: Query executed successfully. No rows returned."

        header = " | ".join(columns)
        rendered_rows = [" | ".join("" if value is None else str(value) for value in row) for row in rows]
        return "\n".join([header, *rendered_rows])

    def _run(self, sql: str) -> str:
        try:
            result = self._conn.execute(sql)
            rows = result.fetchall()
            columns = [desc[0] for desc in (result.description or [])]
            return self._format_result(rows, columns)
        except Exception as exc:  # noqa: BLE001 - intentional passthrough for agent self-correction
            return f"ERROR: {type(exc).__name__}: {exc}"
