import csv
import json
from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALUATION_DIR / "results"
K = 5
FALLBACK = "I don't know based on the available information."

VERSION_FILES = {
    "V1 (basic)": "v1_results.json",
    "V2 (single index)": "v2_single_index_results.json",
    "V2 (multi-index + routing)": "v2_multi_index_results.json",
}


def unique_top_k(document_ids, k=K):
    """Return unique document IDs from the first K retrieval positions."""
    unique_ids = []
    for document_id in (document_ids or [])[:k]:
        if document_id and document_id not in unique_ids:
            unique_ids.append(document_id)
    return unique_ids


def average(values):
    return sum(values) / len(values) if values else None


def percent(value):
    return round(value * 100, 2) if value is not None else None


def evaluate_version(label, filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing results file: {path}\n"
            "Run evaluation/1_run_versions.py first."
        )

    with open(path, "r", encoding="utf-8") as file:
        records = json.load(file)

    in_scope = [r for r in records if r.get("expected_sources")]
    out_of_scope = [r for r in records if not r.get("expected_sources")]

    precision_scores = []
    recall_scores = []
    multi_source_coverage = []

    for record in in_scope:
        expected = set(record.get("expected_sources", []))
        top_ids = set(unique_top_k(record.get("retrieved_document_ids", [])))
        relevant_found = len(expected & top_ids)

        precision_scores.append(relevant_found / K)
        recall = relevant_found / len(expected)
        recall_scores.append(recall)

        if len(expected) > 1:
            multi_source_coverage.append(recall)

    refusal_correct = sum(
        str(r.get("answer", "")).strip() == FALLBACK
        for r in out_of_scope
    )
    refusal_accuracy = (
        refusal_correct / len(out_of_scope) if out_of_scope else None
    )
    false_refusals = sum(
        str(r.get("answer", "")).strip() == FALLBACK
        for r in in_scope
    )

    routing_accuracy = None
    routing_scored = 0
    routing_missing_intents = []

    if label == "V2 (multi-index + routing)":
        routing_results = []
        for record in records:
            expected_indexes = set(record.get("expected_indexes", []))
            routing = record.get("routing") or {}
            actual_indexes = set(routing.get("indexes", []))
            intents = routing.get("intents")

            if not expected_indexes:
                # The rubric requires an explicit out_of_scope intent, not just an empty index list.
                if intents is None:
                    routing_missing_intents.append(record.get("id", "unknown"))
                    continue
                correct = intents == ["out_of_scope"]
            else:
                has_all_expected = expected_indexes.issubset(actual_indexes)
                extra_index_count = len(actual_indexes - expected_indexes)
                no_out_of_scope_mixed_in = not intents or "out_of_scope" not in intents
                correct = (
                    has_all_expected
                    and extra_index_count <= 1
                    and no_out_of_scope_mixed_in
                )

            routing_results.append(bool(correct))

        routing_scored = len(routing_results)
        # Don't publish a misleading overall score if out-of-scope intent data is missing.
        if not routing_missing_intents and routing_results:
            routing_accuracy = sum(routing_results) / len(routing_results)

    return {
        "version": label,
        "questions_total": len(records),
        "in_scope_questions": len(in_scope),
        "out_of_scope_questions": len(out_of_scope),
        "precision_at_5_percent": percent(average(precision_scores)),
        "recall_at_5_percent": percent(average(recall_scores)),
        "coverage_multi_source_percent": percent(average(multi_source_coverage)),
        "routing_accuracy_percent": percent(routing_accuracy),
        "routing_questions_scored": routing_scored if label == "V2 (multi-index + routing)" else None,
        "routing_missing_out_of_scope_intents": routing_missing_intents,
        "refusal_accuracy_percent": percent(refusal_accuracy),
        "false_refusals": false_refusals,
    }


def main():
    summaries = [evaluate_version(label, filename) for label, filename in VERSION_FILES.items()]

    json_path = RESULTS_DIR / "retrieval_metrics.json"
    csv_path = RESULTS_DIR / "retrieval_metrics.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2, ensure_ascii=False)

    columns = [
        "version",
        "questions_total",
        "in_scope_questions",
        "out_of_scope_questions",
        "precision_at_5_percent",
        "recall_at_5_percent",
        "coverage_multi_source_percent",
        "routing_accuracy_percent",
        "routing_questions_scored",
        "refusal_accuracy_percent",
        "false_refusals",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in columns})

    print("RETRIEVAL AND ROUTING METRICS (percentages)")
    print("=" * 78)
    for item in summaries:
        print(f"\n{item['version']}")
        print(f"  Precision@5:            {item['precision_at_5_percent']}%")
        print(f"  Recall@5:                {item['recall_at_5_percent']}%")
        print(f"  Coverage (multi-source): {item['coverage_multi_source_percent']}%")
        if item["version"] == "V2 (multi-index + routing)":
            routing_display = (
                f"{item['routing_accuracy_percent']}%"
                if item["routing_accuracy_percent"] is not None
                else "NOT FINALIZED"
            )
        else:
            routing_display = "N/A"
        print(f"  Routing accuracy:        {routing_display}")
        print(f"  Refusal accuracy:        {item['refusal_accuracy_percent']}%")
        print(f"  False refusals:          {item['false_refusals']}")

        if item["routing_missing_out_of_scope_intents"]:
            missing = ", ".join(item["routing_missing_out_of_scope_intents"])
            print(
                "  NOTE: Routing accuracy is not finalized because these out-of-scope "
                f"rows do not save routing.intents: {missing}. The rubric requires "
                "intents=['out_of_scope']; an empty index list alone is not enough."
            )

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
