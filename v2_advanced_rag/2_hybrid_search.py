import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "data" / "dandes_electronics_kb.json"
VECTORSTORE_DIR = PROJECT_DIR / "chroma_db"

load_dotenv(PROJECT_DIR / ".env")

COLLECTION_NAME = "dandes_v1"
EMBEDDING_MODEL = "text-embedding-3-small"


def load_documents():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)["documents"]


def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def reciprocal_rank_fusion(rankings, k=60):
    scores = {}

    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] = scores.get(document_id, 0) + (
                1 / (k + rank)
            )

    return sorted(scores, key=scores.get, reverse=True)


def print_results(title, results):
    print(f"\n--- {title} ---")

    if not results:
        print("No documents found.")
        return

    for rank, item in enumerate(results, start=1):
        print(
            f"{rank}. Source: {item['id']} | "
            f"Type: {item['doc_type']} | "
            f"Title: {item['title']}"
        )

def hybrid_search(question, top_k=5, candidate_k=20):
    raw_documents = load_documents()

    corpus = [
        f"{item['id']} {item['title']} {item['content']}"
        for item in raw_documents
    ]

    tokenized_corpus = [tokenize(text) for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenize(question))

    bm25_ranking = sorted(
        range(len(raw_documents)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:candidate_k]

    bm25_ids = [raw_documents[i]["id"] for i in bm25_ranking]

    vectorstore = load_vectorstore()
    semantic_results = vectorstore.similarity_search(
        question,
        k=candidate_k,
    )

    semantic_ids = [
        document.metadata["id"] for document in semantic_results
    ]

    documents_by_id = {
        item["id"]: item for item in raw_documents
    }

    before_results = [
        documents_by_id[doc_id]
        for doc_id in semantic_ids[:top_k]
        if doc_id in documents_by_id
    ]

    
    fused_ids = reciprocal_rank_fusion(
        [semantic_ids, bm25_ids]
    )

    question_ids = re.findall(
        r"\b[A-Z]{2,4}-\d{3,4}\b",
        question.upper(),
    )

    for doc_id in question_ids:
        if doc_id in documents_by_id:
            fused_ids = [doc_id] + [
                existing_id
                for existing_id in fused_ids
                if existing_id != doc_id
            ]

    after_results = [
        documents_by_id[doc_id]
        for doc_id in fused_ids[:top_k]
        if doc_id in documents_by_id
    ]

    print("\nQuestion:", question)
    print_results("BEFORE: Semantic Search Top 5", before_results)
    print_results("AFTER: Hybrid Search Top 5", after_results)

    before_ids = {item["id"] for item in before_results}
    after_ids = {item["id"] for item in after_results}
    added_ids = after_ids - before_ids

    print(
        "\nExplanation: Hybrid search combines semantic similarity "
        "with BM25 keyword matching using Reciprocal Rank Fusion. "
        "Exact document IDs in the question are prioritized to "
        "improve retrieval for policy-specific questions."
    )
    print("New IDs appearing in the hybrid top 5:", sorted(added_ids))

    return after_results

if __name__ == "__main__":
    question = input("Enter a customer question: ").strip()

    if question:
        hybrid_search(question)
    else:
        print("Please enter a question.")
