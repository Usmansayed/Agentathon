from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "robust_test_data.csv"
OUTPUT_TXT = ROOT / "robust-output.txt"


def sentence_count(text: str) -> int:
    parts = [p for p in __import__("re").split(r"(?<=[.!?])\s+", text.strip()) if p]
    return len(parts)


def parse_output(path: Path) -> dict[str, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result: dict[str, str] = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def main() -> None:
    start = time.perf_counter()
    command = [
        "python",
        "main.py",
        "--data-path",
        str(INPUT_CSV),
        "--output-file",
        str(OUTPUT_TXT),
        "--fallback-only",
    ]
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start

    checks = {
        "autonomy_no_crash": run.returncode == 0,
        "output_file_created": OUTPUT_TXT.exists(),
        "format_has_q1_to_q5": False,
        "q3_has_5_counts": False,
        "q5_exactly_3_sentences": False,
        "execution_under_15s": elapsed < 15,
    }
    details: dict[str, str] = {"stdout_tail": run.stdout[-600:], "stderr_tail": run.stderr[-600:]}

    if OUTPUT_TXT.exists():
        parsed = parse_output(OUTPUT_TXT)
        checks["format_has_q1_to_q5"] = all(key in parsed for key in ("Q1", "Q2", "Q3", "Q4", "Q5"))
        checks["q3_has_5_counts"] = all(
            label in parsed.get("Q3", "")
            for label in (
                "duplicates=",
                "quantity_outliers=",
                "price_format_errors=",
                "invalid_discounts=",
                "total_nulls=",
            )
        )
        checks["q5_exactly_3_sentences"] = sentence_count(parsed.get("Q5", "")) == 3
        details["output_preview"] = OUTPUT_TXT.read_text(encoding="utf-8")

    passed = sum(1 for ok in checks.values() if ok)
    score = round((passed / len(checks)) * 100, 1)

    report = {
        "hackathon_alignment": {
            "autonomy_and_robustness_25": checks["autonomy_no_crash"] and checks["output_file_created"],
            "output_accuracy_readiness_25": checks["format_has_q1_to_q5"] and checks["q3_has_5_counts"],
            "efficiency_10": checks["execution_under_15s"],
        },
        "checks": checks,
        "score_percent": score,
        "elapsed_seconds": round(elapsed, 3),
        "details": details,
    }

    report_path = ROOT / "robust-test-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Robustness score: {score}% ({passed}/{len(checks)} checks passed)")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
