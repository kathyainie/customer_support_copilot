from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = PROJECT_DIR / "chroma_db"

load_dotenv(PROJECT_DIR / ".env")

COLLECTION_NAME = "dandes_v1"
EMBEDDING_MODEL = "text-embedding-3-small"


def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    return vectorstore


def retrieve_documents(question, k=5):
    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search_with_score(
        question,
        k=k,
    )

    print("\nRetrieved documents:")

    for document, score in results:
        source_id = document.metadata.get("id", "Unknown")
        print(f"\nSource ID: {source_id}")
        print(f"Similarity distance: {score:.4f}")
        print(document.page_content)

    return results


if __name__ == "__main__":
    question = input("Enter a customer question: ")
    retrieve_documents(question)
