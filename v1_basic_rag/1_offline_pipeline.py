import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "data" / "dandes_electronics_kb.json"
VECTORSTORE_DIR = PROJECT_DIR / "chroma_db"


load_dotenv(PROJECT_DIR / ".env")

COLLECTION_NAME = "dandes_v1"
EMBEDDING_MODEL = "text-embedding-3-small"


def load_documents():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["documents"]




def prepare_documents(raw_documents):
    documents = []

    for item in raw_documents:
        metadata = {
            key: value
            for key, value in item["metadata"].items()
            if value is not None
        }

        metadata["id"] = item["id"]
        metadata["doc_type"] = item["doc_type"]

        # One JSON record = one Chroma document; no chunking
        page_content = f"{item['title']}\n\n{item['content']}"

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata,
            )
        )

    return documents


def build_index():
    raw_documents = load_documents()
    documents = prepare_documents(raw_documents)

    print(f"Documents prepared: {len(documents)}")

    if len(documents) != 50:
        raise ValueError(
            f"Expected exactly 50 documents, but found {len(documents)}."
        )

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

   
    document_ids = [document.metadata["id"] for document in documents]
    vectorstore.add_documents(documents, ids=document_ids)

    stored_count = vectorstore._collection.count()
    print(f"Chroma collection: {COLLECTION_NAME}")
    print(f"Documents stored: {stored_count}")
    print(f"Vector database folder: {VECTORSTORE_DIR}")

if __name__ == "__main__":
    build_index()
