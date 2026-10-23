
import pickle
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder


PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = PROJECT_DIR / "chroma_db"
BM25_DIR = Path(__file__).resolve().parent / "bm25_indexes"

load_dotenv(PROJECT_DIR / ".env")

EMBEDDING_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INDEX_NAMES = [
    "policy_index",
    "product_index",
    "order_index",
    "ticket_index",
]

TOP_K_DENSE = 10
TOP_K_BM25 = 10
TOP_K_FINAL = 5
RRF_K = 60


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def reciprocal_rank_fusion(result_lists):
    scores = {}
    documents = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            doc_id = doc.metadata.get("id")

            if not doc_id:
                continue

            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + 1.0 / (RRF_K + rank)
            )
            documents[doc_id] = doc

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [documents[doc_id] for doc_id in ranked_ids]


def load_index(index_name, embeddings):
    bm25_path = BM25_DIR / f"{index_name}.pkl"

    if not bm25_path.exists():
        raise FileNotFoundError(
            f"BM25 index missing: {bm25_path}\n"
            "Run 1_build_indexes.py first."
        )

    with open(bm25_path, "rb") as file:
        bm25_data = pickle.load(file)

    store = Chroma(
        collection_name=index_name,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    return (
        store,
        bm25_data["bm25"],
        bm25_data["documents"],
    )


def hybrid_search(query, index_name, loaded_index):
    store, bm25, documents = loaded_index

    dense_results = store.similarity_search(
        query,
        k=TOP_K_DENSE,
    )

    bm25_scores = bm25.get_scores(tokenize(query))

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: float(bm25_scores[i]),
        reverse=True,
    )[:TOP_K_BM25]

    bm25_results = [
        documents[i] for i in bm25_indices
    ]

    return reciprocal_rank_fusion([
        dense_results,
        bm25_results,
    ])


def global_rerank(query, candidates, reranker):
    if not candidates:
        return []

    pairs = [
        (query, doc.page_content)
        for doc in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    print("\nGLOBAL CROSS-ENCODER RERANKING — TOP 15")

    for rank, (doc, score) in enumerate(
        ranked[:15], start=1
    ):
        print(
            f"{rank}. {doc.metadata.get('id')} "
            f"| score={float(score):.4f} "
            f"| {doc.metadata.get('title', 'Untitled')}"
        )
    candidate_by_id = {
        doc.metadata.get("id"): doc
        for doc in candidates
        if doc.metadata.get("id")
    }

    selected = [
        doc for doc, score in ranked[:TOP_K_FINAL]
    ]

    reference_pattern = re.compile(
        r"\b(?:RR|SH|OG|KI|FAQ|TCK)-\d{3,4}\b"
    )

    for source_doc, score in ranked[:3]:
        source_metadata = source_doc.metadata or {}
        source_id = source_metadata.get("id")
        source_type = source_metadata.get("doc_type")

        references = reference_pattern.findall(
            source_doc.page_content
        )

        for ref_id in references:
            selected_ids = {
                (doc.metadata or {}).get("id")
                for doc in selected
            }

            if ref_id not in candidate_by_id:
                continue

            if ref_id in selected_ids:
                continue

            linked_doc = candidate_by_id[ref_id]

            replace_index = None

            for i in range(len(selected) - 1, -1, -1):
                current_metadata = selected[i].metadata or {}

                if (
                    current_metadata.get("doc_type") == source_type
                    and current_metadata.get("id") != source_id
                ):
                    replace_index = i
                    break

            if replace_index is None:
                replace_index = len(selected) - 1

            if selected:
                selected[replace_index] = linked_doc

    print("\nFINAL TOP 5 — AFTER CROSS-REFERENCE CHECK")

    for rank, doc in enumerate(selected, start=1):
        print(
            f"{rank}. {doc.metadata.get('id')} "
            f"| {doc.metadata.get('title', 'Untitled')}"
        )

    return selected

def search_selected_indexes(
    query,
    selected_indexes,
    loaded_indexes,
    reranker,
):
    result_lists = []

    for index_name in selected_indexes:
        results = hybrid_search(
            query,
            index_name,
            loaded_indexes[index_name],
        )

        print(
            f"{index_name}: "
            f"{[doc.metadata.get('id') for doc in results[:5]]}"
        )

        result_lists.append(results)

    candidates = reciprocal_rank_fusion(result_lists)

    print(f"\nUnique merged candidates: {len(candidates)}")

    return global_rerank(
        query,
        candidates,
        reranker,
    )


def main():
    query = (
        "A customer wants to change the address after the order "
        "shipped. What is the official process, and do we have "
        "a resolved case like this?"
    )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )
    reranker = CrossEncoder(RERANKER_MODEL)

    loaded_indexes = {
        name: load_index(name, embeddings)
        for name in INDEX_NAMES
    }

    all_results = search_selected_indexes(
        query,
        INDEX_NAMES,
        loaded_indexes,
        reranker,
    )

    print("\nTOP 5 — ALL INDEXES")
    for doc in all_results:
        print(doc.metadata.get("id"), doc.metadata.get("title"))

    focused_indexes = ["order_index", "ticket_index"]

    focused_results = search_selected_indexes(
        query,
        focused_indexes,
        loaded_indexes,
        reranker,
    )

    print("\nTOP 5 — ORDER + TICKET INDEXES")
    for doc in focused_results:
        print(doc.metadata.get("id"), doc.metadata.get("title"))


if __name__ == "__main__":
    main()
