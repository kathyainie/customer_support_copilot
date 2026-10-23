from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = PROJECT_DIR / "chroma_db"

load_dotenv(PROJECT_DIR / ".env")

COLLECTION_NAME = "dandes_v1"
EMBEDDING_MODEL = "text-embedding-3-small"

VALID_DOC_TYPES = {
    "product_faq",
    "return_policy",
    "shipping_policy",
    "resolved_ticket",
    "known_issue",
    "order_guide",
}


def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )


def print_results(title, results):
    print(f"\n--- {title} ---")

    if not results:
        print("No documents found.")
        return

    for rank, (document, score) in enumerate(results, start=1):
        source_id = document.metadata.get("id", "Unknown")
        doc_type = document.metadata.get("doc_type", "Unknown")

        print(
            f"\n{rank}. Source: {source_id} | "
            f"Type: {doc_type} | Distance: {score:.4f}"
        )
        print(document.page_content)


def metadata_filtered_search(question, doc_type=None, k=5):
    vectorstore = load_vectorstore()

    before_results = vectorstore.similarity_search_with_score(
        question,
        k=k,
    )
    
    after_results = before_results

    if doc_type:
        after_results = vectorstore.similarity_search_with_score(
            question,
            k=k,
            filter={"doc_type": doc_type},
        )

    print("\nQuestion:", question)
    print("Metadata filter:", doc_type or "None")

    print_results("BEFORE FILTERING (top 5)", before_results)
    print_results("AFTER FILTERING (top 5)", after_results)

    before_ids = {
        document.metadata.get("id")
        for document, _ in before_results
    }
    after_ids = {
        document.metadata.get("id")
        for document, _ in after_results
    }

    if not doc_type:
        print("\nExplanation: No filter was selected.")
    else:
        print(
            "\nExplanation: Metadata filtering restricts retrieval "
            f"to documents with doc_type='{doc_type}'. "
            "This can exclude irrelevant document categories."
        )

        removed_ids = before_ids - after_ids
        print("Previously retrieved IDs excluded by the filter:",
              sorted(removed_ids))

    return before_results, after_results


if __name__ == "__main__":
    question = input("Enter a customer question: ").strip()

    if not question:
        print("Please enter a question.")
    else:
        doc_type = input(
            "Filter by type "
            "(product_faq, return_policy, shipping_policy, "
            "resolved_ticket, known_issue, order_guide), "
            "or press Enter for no filter: "
        ).strip()

        if doc_type and doc_type not in VALID_DOC_TYPES:
            print("Invalid document type. Please use one of the listed types.")
        else:
            metadata_filtered_search(
                question,
                doc_type=doc_type or None,
            )
