import argparse
import contextlib
import importlib.util
import io
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_DIR / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"
GOLDEN_SET_PATH = EVALUATION_DIR / "golden_test_set.json"

FALLBACK = "I don't know based on the available information."

load_dotenv(PROJECT_DIR / ".env")


def load_module(relative_path, module_name):
    """Import an existing project script without running its main function."""
    path = PROJECT_DIR / relative_path

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python file: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

v1 = load_module(
    "v1_basic_rag/2_online_pipeline.py",
    "evaluation_v1_online",
)

v2 = load_module(
    "v2_advanced_rag/5_v2_pipeline.py",
    "evaluation_v2_advanced",
)

multi = load_module(
    "v2_multi_index/4_v2_multi_index_pipeline.py",
    "evaluation_v2_multi_index",
)


def response_text(content):
    """Convert a model response to plain text."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        ).strip()

    return str(content).strip()


def parse_json_response(text):
    """Parse a JSON object from a model response."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = (
            cleaned.replace("```json", "", 1)
            .replace("```", "")
            .strip()
        )

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end <= start:
        raise ValueError("No JSON object found in the model response.")

    return json.loads(cleaned[start:end + 1])


def unique_ids(values):
    """Remove duplicates while keeping the original order."""
    return list(dict.fromkeys(value for value in values if value))


def make_record(question_data, version, answer, retrieved_ids, sources, routing):
    return {
        "id": question_data["id"],
        "version": version,
        "question": question_data["question"],
        "reference_answer": question_data["reference_answer"],
        "expected_sources": question_data["expected_sources"],
        "expected_indexes": question_data["expected_indexes"],
        "answer": answer,
        "retrieved_document_ids": unique_ids(retrieved_ids),
        "sources": unique_ids(sources),
        "routing": routing,
    }


def generate_v1_answer(question, documents, answer_model):
    """Generate an answer using the documents retrieved by the V1 pipeline."""
    if not documents:
        return FALLBACK, []

    available_ids = [
        doc.metadata.get("id")
        for doc in documents
        if doc.metadata.get("id")
    ]

    context = "\n\n".join(
        (
            f"Source ID: {doc.metadata.get('id', 'Unknown')}\n"
            f"Title: {doc.metadata.get('title', 'Untitled')}\n"
            f"Content: {doc.page_content}"
        )
        for doc in documents
    )

    prompt = f"""
You are the Dandes Electronics Customer Support Copilot.

Answer the question using only the supplied knowledge-base documents.
Do not invent facts. If the documents do not support an answer, use
exactly this answer:
{FALLBACK}

Return only valid JSON with these fields:
{{
  "answer": "Your answer here",
  "sources": ["SOURCE-ID"]
}}

Cite only IDs in the supplied context. Use an empty sources list
if the answer is the fallback.

Question:
{question}

Available source IDs:
{available_ids}

Knowledge-base documents:
{context}
"""

    response = answer_model.invoke(prompt)
    raw_text = response_text(response.content)

    try:
        result = parse_json_response(raw_text)
        answer = str(result.get("answer", "")).strip()
        proposed_sources = result.get("sources", [])

        if not answer:
            raise ValueError("The model returned an empty answer.")

        if answer == FALLBACK:
            return FALLBACK, []

        allowed = set(available_ids)
        sources = unique_ids(
            [
                source.strip()
                for source in proposed_sources
                if isinstance(source, str) and source.strip() in allowed
            ]
        )

        return answer, sources

    except (ValueError, json.JSONDecodeError, TypeError):
        return raw_text or FALLBACK, []


def run_v1(question_data, answer_model):
    question = question_data["question"]

    with contextlib.redirect_stdout(io.StringIO()):
        retrieved = v1.retrieve_documents(question, k=5)

    documents = [document for document, score in retrieved]
    retrieved_ids = [
        document.metadata.get("id")
        for document in documents
        if document.metadata.get("id")
    ]

    answer, sources = generate_v1_answer(
        question,
        documents,
        answer_model,
    )

    return make_record(
    question_data,
    "v1_basic_rag",
    answer,
    retrieved_ids,
    sources,
    {"intents": [], "indexes": ["dandes_v1"]},
)
    


def extract_sources_from_answer(answer_text, retrieved_ids):
    """Extract cited IDs from a V2 Advanced answer's Sources line."""
    match = re.search(
            r"(?is)\bSources:\s*(.*?)\s*$",
        answer_text,
    )

    if not match:
        return []

    proposed_ids = re.findall(
        r"\b(?:RR|SH|OG|FAQ|KI|TCK)-\d+\b",
        match.group(1),
    )

    allowed = set(retrieved_ids)
    return unique_ids([source for source in proposed_ids if source in allowed])


def clean_answer(answer_text):
    """Remove the separate Sources line from the answer field."""
    if FALLBACK in answer_text:
        return FALLBACK

    return re.sub(
        r"(?is)\s*\bSources:\s*.*$",
        "",
        answer_text,
    ).strip()


def run_v2_advanced(question_data):
    question = question_data["question"]

    with contextlib.redirect_stdout(io.StringIO()):
        documents = v2.advanced_search(
            question,
            doc_type=None,
            candidate_k=20,
            top_k=5,
        )

    retrieved_ids = [
        item.get("id")
        for item in documents
        if item.get("id")
    ]

    with contextlib.redirect_stdout(io.StringIO()):
        answer_text = response_text(
            v2.generate_answer(question, documents)
        )

    sources = extract_sources_from_answer(answer_text, retrieved_ids)
    answer = clean_answer(answer_text)

    return make_record(
    question_data,
    "v2_advanced_rag",
    answer or FALLBACK,
    retrieved_ids,
    sources,
    {"intents": [], "indexes": ["dandes_v1"]},
)
    


def run_v2_multi_index(question_data, loaded_indexes, reranker, answer_model):
    question = question_data["question"]
    last_retrieved_ids = []
    original_search = multi.search_module.search_selected_indexes

    def tracked_search(*args, **kwargs):
        nonlocal last_retrieved_ids

        documents = original_search(*args, **kwargs)

        last_retrieved_ids = [
            doc.metadata.get("id")
            for doc in documents
            if doc.metadata.get("id")
        ]
        return documents

    multi.search_module.search_selected_indexes = tracked_search

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            item = multi.run_question(
                question,
                loaded_indexes,
                reranker,
                answer_model,
            )
    finally:
        multi.search_module.search_selected_indexes = original_search

    return make_record(
        question_data,
        "v2_multi_index_rag",
        item.get("answer", FALLBACK),
        last_retrieved_ids,
        item.get("sources", []),
         {
            "intents": item.get("intents", []),
            "indexes": item.get("indexes", []),
        },
    )


def save_results(filename, results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the Customer Support Copilot evaluation."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions as a quick test.",
    )
    args = parser.parse_args()

    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(
            f"Golden test set not found: {GOLDEN_SET_PATH}"
        )

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list) or not questions:
        raise ValueError("golden_test_set.json must contain a non-empty JSON list.")

    required_fields = {
        "id",
        "question",
        "reference_answer",
        "expected_sources",
        "expected_indexes",
    }

    for item in questions:
        missing = required_fields - item.keys()
        if missing:
            raise ValueError(
                f"{item.get('id', 'Unknown question')} is missing fields: "
                f"{sorted(missing)}"
            )

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        questions = questions[:args.limit]

    print("=" * 72)
    print("CUSTOMER SUPPORT COPILOT — EVALUATION RUNNER")
    print(f"Questions to evaluate: {len(questions)}")
    print("=" * 72)

   
    answer_model = v2.ChatOpenAI(model=v2.CHAT_MODEL)

    
    print("\n[1/3] Running V1 Basic RAG...")
    v1_results = []

    for number, question_data in enumerate(questions, start=1):
        print(f"  V1: {number}/{len(questions)} — {question_data['id']}")
        v1_results.append(run_v1(question_data, answer_model))

    save_results("v1_results.json", v1_results)

    print("\n[2/3] Running V2 Advanced RAG...")
    v2_results = []

    for number, question_data in enumerate(questions, start=1):
        print(f"  V2 Advanced: {number}/{len(questions)} — {question_data['id']}")
        v2_results.append(run_v2_advanced(question_data))

    save_results("v2_single_index_results.json", v2_results)

   
    print("\n[3/3] Loading the multi-index models and indexes...")

    embeddings = multi.OpenAIEmbeddings(
        model=multi.EMBEDDING_MODEL
    )
    reranker = multi.CrossEncoder(multi.RERANKER_MODEL)
    multi_answer_model = multi.ChatOpenAI(model=multi.CHAT_MODEL)

    loaded_indexes = {
        name: multi.search_module.load_index(name, embeddings)
        for name in multi.search_module.INDEX_NAMES
    }

    print("Running V2 Multi-Index RAG...")
    multi_results = []

    for number, question_data in enumerate(questions, start=1):
        print(
            f"  V2 Multi-Index: {number}/{len(questions)} — "
            f"{question_data['id']}"
        )
        multi_results.append(
            run_v2_multi_index(
                question_data,
                loaded_indexes,
                reranker,
                multi_answer_model,
            )
        )

    save_results("v2_multi_index_results.json", multi_results)

    print("\n" + "=" * 72)
    print("EVALUATION RUN COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()