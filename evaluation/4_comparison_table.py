import csv
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_DIR / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"
OUTPUT_PATH = EVALUATION_DIR / "comparison_table.csv"

VERSION_LABELS = [
    "V1 (basic)",
    "V2 (single index)",
    "V2 (multi-index + routing)",
]


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required metrics file not found: {path}\n"
            "Run 2_retrieval_metrics.py and 3_generation_metrics.py first."
        )
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def index_by_version(rows):
    return {row.get("version"): row for row in rows}


def fmt_percent(value):
    return "N/A" if value is None or value == "" else f"{float(value):.2f}%"


def fmt_score(value):
    return "N/A" if value is None or value == "" else f"{float(value):.3f}"


def main():
    retrieval = index_by_version(load_json(RESULTS_DIR / "retrieval_metrics.json"))
    generation = index_by_version(load_json(RESULTS_DIR / "generation_metrics.json"))

    rows = []
    for metric, retrieval_key, generation_key, formatter in [
        ("Routing Accuracy", "routing_accuracy_percent", None, fmt_percent),
        ("Precision@5", "precision_at_5_percent", None, fmt_percent),
        ("Recall@5", "recall_at_5_percent", None, fmt_percent),
        ("Coverage (multi-source)", "coverage_multi_source_percent", None, fmt_percent),
        ("Faithfulness (LLM judge / RAGAS)", None, "faithfulness_pair", str),
        ("Answer Relevancy", None, "answer_relevancy_mean", fmt_score),
        ("Context Precision / Recall", None, "context_pair", str),
        ("Refusal Accuracy", "refusal_accuracy_percent", None, fmt_percent),
        ("False Refusals", "false_refusals", None, lambda x: "N/A" if x is None else str(x)),
    ]:
        output_row = {"Metric": metric}
        for version in VERSION_LABELS:
            r = retrieval.get(version, {})
            g = generation.get(version, {})
            if metric == "Faithfulness (LLM judge / RAGAS)":
                judge = g.get("llm_judge_faithfulness_mean")
                ragas = g.get("ragas_faithfulness_mean")
                value = "N/A" if judge is None and ragas is None else (
                    f"Judge {fmt_score(judge)} / RAGAS {fmt_score(ragas)}"
                )
            elif metric == "Context Precision / Recall":
                precision = g.get("context_precision_mean")
                recall = g.get("context_recall_mean")
                value = "N/A" if precision is None and recall is None else (
                    f"{fmt_score(precision)} / {fmt_score(recall)}"
                )
            elif retrieval_key is not None:
                value = formatter(r.get(retrieval_key))
            else:
                value = formatter(g.get(generation_key))
            # Routing is not applicable to either single-index version.
            if metric == "Routing Accuracy" and version != "V2 (multi-index + routing)":
                value = "N/A"
            output_row[version] = value
        rows.append(output_row)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Metric"] + VERSION_LABELS,
        )
        writer.writeheader()
        writer.writerows(rows)

    print("FINAL VERSION COMPARISON")
    print("=" * 78)
    for row in rows:
        print(f"{row['Metric']}")
        for version in VERSION_LABELS:
            print(f"  {version}: {row[version]}")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
