import argparse
import asyncio
import csv
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_DIR / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"
DATASET_PATH = PROJECT_DIR / "data" / "dandes_electronics_kb.json"
FALLBACK = "I don't know based on the available information."
K = 5

VERSION_FILES = {
    "V1 (basic)": "v1_results.json",
    "V2 (single index)": "v2_single_index_results.json",
    "V2 (multi-index + routing)": "v2_multi_index_results.json",
}

load_dotenv(PROJECT_DIR / ".env")

JUDGE_MODEL = os.getenv("EVALUATION_JUDGE_MODEL", "gpt-6-luna")
EMBEDDING_MODEL = os.getenv(
    "EVALUATION_EMBEDDING_MODEL", "text-embedding-3-small"
)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run evaluation/1_run_versions.py after the final code changes."
        )
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_document_map():
    dataset = load_json(DATASET_PATH)
    documents = dataset.get("documents", []) if isinstance(dataset, dict) else dataset
    if not isinstance(documents, list):
        raise ValueError("The knowledge-base JSON must contain a documents list.")
    mapping = {}
    for doc in documents:
        if not isinstance(doc, dict) or not doc.get("id"):
            continue
        mapping[str(doc["id"])] = doc
    return mapping


def retrieved_contexts(record, document_map):
    """Rebuild the exact retrieved context from the stored top-five IDs."""
    contexts = []
    seen = set()
    for doc_id in (record.get("retrieved_document_ids") or [])[:K]:
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        doc = document_map.get(str(doc_id))
        if not doc:
            continue
        contexts.append(
            f"Document ID: {doc_id}\n"
            f"Title: {doc.get('title', 'Untitled')}\n"
            f"Document type: {doc.get('doc_type', 'document')}\n"
            f"Content:\n{doc.get('content', '')}"
        )
    return contexts


def to_float(value):
    if hasattr(value, "value"):
        value = value.value
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return max(0.0, min(1.0, number))


def parse_json_response(text):
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1).replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Judge response did not contain a JSON object.")
    return json.loads(cleaned[start:end + 1])


def llm_judge_faithfulness(judge, question, answer, contexts):
    """Return a rubric-based 0-1 groundedness score from the configured judge."""
    context_text = "\n\n---\n\n".join(contexts)
    prompt = f"""
You are a strict evaluation judge for a customer-support RAG system.
Score ONLY factual faithfulness: whether factual claims in the generated answer
are supported by the supplied retrieved context. Do not score helpfulness,
coverage, writing style, or answer relevance.

Use this rubric:
- 1.0: every factual claim is supported by the context.
- 0.5: some factual claims are supported and some are unsupported or contradicted.
- 0.0: the answer's factual claims are mostly unsupported or contradicted.
- For an answer that is exactly the refusal sentence "{FALLBACK}", return a score
  of 0.0 here because it is not an informative answer to this in-scope question.
- Do not treat a missing detail as a factual error unless the answer asserts it.

Return JSON only, with score (number from 0 to 1) and a short explanation.

Question:
{question}

Generated answer:
{answer}

Retrieved context:
{context_text}
"""
    response = judge.invoke(prompt)
    parsed = parse_json_response(getattr(response, "content", response))
    score = to_float(parsed.get("score"))
    if score is None:
        raise ValueError("LLM judge did not return a numeric score.")
    return score, str(parsed.get("explanation", "")).strip()


def average(values):
    usable = [value for value in values if value is not None]
    return round(sum(usable) / len(usable), 4) if usable else None


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


async def score_ragas(scorers, question, answer, contexts, reference):
    """Run the current RAGAS collections API metrics for one test case."""
    output = {}
    for key, scorer in scorers.items():
        if key in {"context_precision", "context_recall"}:
            kwargs = {
                "user_input": question,
                "retrieved_contexts": contexts,
                "reference": reference,
            }
        elif key == "answer_relevancy":
            kwargs = {
                "user_input": question,
                "response": answer,
            }
        else:  # RAGAS faithfulness
            kwargs = {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
            }
        result = await scorer.ascore(**kwargs)
        output[key] = to_float(result)
    return output


async def main_async(args):
    
    try:
        from openai import AsyncOpenAI
        from langchain_openai import ChatOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings.base import embedding_factory
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
        )
    except ImportError as error:
        raise ImportError(
            "Generation metrics require the current RAGAS API and OpenAI/LangChain "
            "packages. In the active .venv, install/update with: "
            "pip install -U ragas openai langchain-openai python-dotenv"
        ) from error

    document_map = load_document_map()
    client = AsyncOpenAI()
    ragas_llm = llm_factory(
    JUDGE_MODEL,
    client=client,
    max_tokens=8192,
)
    ragas_embeddings = embedding_factory(
        "openai", model=EMBEDDING_MODEL, client=client
    )
    scorers = {
        "ragas_faithfulness": Faithfulness(llm=ragas_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=ragas_llm, embeddings=ragas_embeddings
        ),
        "context_precision": ContextPrecision(llm=ragas_llm),
        "context_recall": ContextRecall(llm=ragas_llm),
    }
    judge = ChatOpenAI(model=JUDGE_MODEL,  max_retries=2)

    per_question_path = RESULTS_DIR / "generation_metrics_per_question.json"
    if args.force or not per_question_path.exists():
        per_question = []
    else:
        per_question = load_json(per_question_path)

    cached = {
        (item.get("version"), item.get("id")): item
        for item in per_question
        if isinstance(item, dict) and item.get("version") and item.get("id")
    }

    total_new = 0
    for version, filename in VERSION_FILES.items():
        records = load_json(RESULTS_DIR / filename)
        in_scope = [record for record in records if record.get("expected_sources")]
        if args.limit is not None:
            in_scope = in_scope[:args.limit]

        for record in in_scope:
            key = (version, record.get("id"))
            if not args.force and key in cached:
                print(f"Cached: {version} — {record.get('id')}")
                continue

            question = str(record.get("question", ""))
            answer = str(record.get("answer", ""))
            reference = str(record.get("reference_answer", ""))
            contexts = retrieved_contexts(record, document_map)

            result = {
                "id": record.get("id"),
                "version": version,
                "question": question,
                "llm_judge_faithfulness": None,
                "llm_judge_explanation": "",
                "ragas_faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
                "scoring_note": "",
            }

            if not contexts:
                result["scoring_note"] = (
                    "No retrieved context could be reconstructed from the dataset IDs; "
                    "generation metrics were left blank for this question."
                )
            else:
                judge_score, explanation = llm_judge_faithfulness(
                    judge, question, answer, contexts
                )
                result["llm_judge_faithfulness"] = judge_score
                result["llm_judge_explanation"] = explanation

                ragas_scores = await score_ragas(
                    scorers, question, answer, contexts, reference
                )
                result.update(ragas_scores)

            cached[key] = result
            per_question = list(cached.values())
            save_json(per_question_path, per_question)  
            total_new += 1
            print(
                f"Scored {version}: {record.get('id')} "
                f"({total_new} new question(s) this run)"
            )

    metric_keys = [
        "llm_judge_faithfulness",
        "ragas_faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    summaries = []
    for version in VERSION_FILES:
        rows = [item for item in cached.values() if item.get("version") == version]
        summary = {
            "version": version,
            "questions_scored": len(rows),
        }
        for metric in metric_keys:
            summary[f"{metric}_mean"] = average(
                [to_float(row.get(metric)) for row in rows]
            )
        summaries.append(summary)

    summary_json_path = RESULTS_DIR / "generation_metrics.json"
    summary_csv_path = RESULTS_DIR / "generation_metrics.csv"
    per_question_csv_path = RESULTS_DIR / "generation_metrics_per_question.csv"
    save_json(summary_json_path, summaries)
    summary_columns = ["version", "questions_scored"] + [
        f"{metric}_mean" for metric in metric_keys
    ]
    save_csv(summary_csv_path, summaries, summary_columns)
    save_csv(
        per_question_csv_path,
        sorted(cached.values(), key=lambda row: (row["version"], row["id"])),
        [
            "id", "version", "question", "llm_judge_faithfulness",
            "llm_judge_explanation", "ragas_faithfulness", "answer_relevancy",
            "context_precision", "context_recall", "scoring_note",
        ],
    )

    print("\nGENERATION METRICS (0 to 1; higher is better)")
    print("=" * 78)
    for summary in summaries:
        print(f"\n{summary['version']} — n={summary['questions_scored']}")
        for metric in metric_keys:
            print(f"  {metric}: {summary[f'{metric}_mean']}")
    print(f"\nSaved: {summary_json_path}")
    print(f"Saved: {summary_csv_path}")
    print(f"Saved: {per_question_csv_path}")
    await client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Score generation quality with an LLM judge and RAGAS."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Score only the first N in-scope questions per version for a test run.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-score all selected questions instead of reusing saved checkpoints.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
