import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder

PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = PROJECT_DIR / "chroma_db"
DATASET_PATH = PROJECT_DIR / "data" / "dandes_electronics_kb.json"

load_dotenv(PROJECT_DIR / ".env")

COLLECTION_NAME = "dandes_v1"
EMBEDDING_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_titles():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    return {
        doc["id"]: doc["title"]
        for doc in data["documents"]
    }


def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )


def rerank_documents(question, candidate_k=20, top_k=5):
    titles = load_titles()
    vectorstore = load_vectorstore()

    candidates = vectorstore.similarity_search(
        question,
        k=candidate_k,
    )

    if not candidates:
        print("No documents were retrieved.")
        return []

    reranker = CrossEncoder(RERANKER_MODEL)

    pairs = [
        (question, document.page_content)
        for document in candidates
    ]

    scores = reranker.predict(pairs)

    ranked_documents = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    final_results = ranked_documents[:top_k]

    print("\nBefore reranking (Top 5):")
    for rank, document in enumerate(candidates[:5], start=1):
        source_id = document.metadata.get("id", "Unknown")
        print(f"\nRank: {rank}")
        print(f"Source ID: {source_id}")
        print(f"Title: {titles.get(source_id, 'Unknown')}")
        print(f"Content: {document.page_content}")

    print(
        "\nExplanation: The cross-encoder scores each question-document "
        "pair together, then reorders the retrieved candidates by relevance."
    )

    print("\nAfter reranking (Top 5):")
    for rank, (document, score) in enumerate(final_results, start=1):
        source_id = document.metadata.get("id", "Unknown")
        print(f"\nRank: {rank}")
        print(f"Source ID: {source_id}")
        print(f"Title: {titles.get(source_id, 'Unknown')}")
        print(f"Reranker score: {float(score):.4f}")
        print(f"Content: {document.page_content}")

    return [document for document, score in final_results]


if __name__ == "__main__":
    question = input("Enter a customer question: ")
    rerank_documents(question)
