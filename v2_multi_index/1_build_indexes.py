
import json
import pickle
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "data" / "dandes_electronics_kb.json"
VECTORSTORE_DIR = PROJECT_DIR / "chroma_db"
BM25_DIR = Path(__file__).resolve().parent / "bm25_indexes"

load_dotenv(PROJECT_DIR / ".env")

EMBEDDING_MODEL = "text-embedding-3-small"

TYPE_TO_INDEX = {
    "return_policy": "policy_index",
    "shipping_policy": "policy_index",
    "product_faq": "product_index",
    "known_issue": "product_index",
    "order_guide": "order_index",
    "resolved_ticket": "ticket_index",
}

EXPECTED_COUNTS = {
    "policy_index": 10,
    "product_index": 20,
    "order_index": 6,
    "ticket_index": 14,
}


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    raw_documents = data.get("documents", [])

    grouped = {
        name: [] for name in EXPECTED_COUNTS
    }

    for item in raw_documents:
        index_name = TYPE_TO_INDEX.get(item.get("doc_type"))

        if index_name is None:
            print("Skipping unknown type:", item.get("doc_type"))
            continue

        metadata = {
            "id": str(item.get("id", "")),
            "doc_type": str(item.get("doc_type", "")),
            "title": str(item.get("title", "")),
        }

        extra = item.get("metadata", {})
        if isinstance(extra, dict):
            for key, value in extra.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[str(key)] = value

        content = (
            f"Title: {item.get('title', '')}\n"
            f"Content: {item.get('content', '')}"
        )

        grouped[index_name].append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    for index_name, expected in EXPECTED_COUNTS.items():
        actual = len(grouped[index_name])
        print(f"{index_name}: {actual} documents")

        if actual != expected:
            raise ValueError(
                f"{index_name}: expected {expected}, got {actual}"
            )

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    BM25_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    for index_name, documents in grouped.items():
        print(f"\nBuilding {index_name}...")

        # Rebuild only this named collection.
        store = Chroma(
            collection_name=index_name,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR),
        )

        try:
            store.delete_collection()
        except Exception:
            pass

        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=index_name,
            persist_directory=str(VECTORSTORE_DIR),
        )

        bm25 = BM25Okapi([
            tokenize(doc.page_content)
            for doc in documents
        ])

        bm25_path = BM25_DIR / f"{index_name}.pkl"

        with open(bm25_path, "wb") as file:
            pickle.dump(
                {
                    "bm25": bm25,
                    "documents": documents,
                },
                file,
            )

        print(f"Saved {len(documents)} documents.")

    print("\nAll four indexes built successfully.")


if __name__ == "__main__":
    main()
