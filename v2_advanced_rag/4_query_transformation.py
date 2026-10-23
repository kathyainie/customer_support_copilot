
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = PROJECT_DIR / "chroma_db"

load_dotenv(PROJECT_DIR / ".env")

COLLECTION_NAME = "dandes_v1"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-6-luna"


def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )


def clean_queries(response_text):
    queries = []

    for line in response_text.splitlines():
        line = re.sub(r"^\s*(?:\d+[\.\)\-]|[-*])\s*", "", line)
        line = line.strip().strip('"').strip("'")

        if line:
            queries.append(line)

    return queries


def rewrite_query(question):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Rewrite the customer's question as ONE clear search query
for a customer support knowledge base.
Preserve product names, order numbers, policy codes, and key details.
Return only the rewritten query, without numbering or explanation."""
        ),
        ("human", "{question}"),
    ])

    model = ChatOpenAI(model=CHAT_MODEL)
    response = (prompt | model).invoke({"question": question})

    return response.content.strip()


def generate_multiple_queries(question):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Generate exactly THREE alternative search queries for
the customer's question in a customer support knowledge base.

Preserve the original intent, product names, policy codes, problem,
and requested resolution. Use meaningfully different phrasings.
Do not introduce unrelated products or change the topic.
Return one query per line, without numbering or explanations."""
        ),
        ("human", "{question}"),
    ])

    model = ChatOpenAI(model=CHAT_MODEL)
    response = (prompt | model).invoke({"question": question})

    return clean_queries(response.content)[:3]


def reciprocal_rank_fusion(rankings, k=60):
    scores = {}

    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] = scores.get(document_id, 0) + (
                1 / (k + rank)
            )

    return sorted(scores, key=scores.get, reverse=True)


def print_results(title, documents):
    print(f"\n--- {title} ---")

    if not documents:
        print("No documents found.")
        return

    for rank, document in enumerate(documents, start=1):
        print(
            f"{rank}. Source: {document.metadata.get('id', 'Unknown')} "
            f"| Type: {document.metadata.get('doc_type', 'Unknown')}"
        )
        print(f"Title/content: {document.page_content}\n")


def get_ranked_ids(vectorstore, query, k=20):
    results = vectorstore.similarity_search(query, k=k)

    ids = []
    documents_by_id = {}

    for document in results:
        source_id = document.metadata.get("id")

        if source_id:
            ids.append(source_id)
            documents_by_id[source_id] = document

    return ids, documents_by_id


def run_query_rewriting(question, vectorstore, k=5):
    print("\n========== DEMO A: QUERY REWRITING ==========")

    original = vectorstore.similarity_search(question, k=k)
    rewritten_query = rewrite_query(question)
    rewritten = vectorstore.similarity_search(rewritten_query, k=k)

    print("\nOriginal question:", question)
    print("Rewritten query:", rewritten_query)

    print_results("BEFORE: Original Query Top 5", original)
    print_results("AFTER: Rewritten Query Top 5", rewritten)

    print(
        "\nExplanation: Query rewriting reformulates the customer's "
        "question into a clearer search query. The before-and-after "
        "results show whether the new wording improves retrieval."
    )


def run_multi_query(question, vectorstore, k=5):
    print(
        "\n========== DEMO B: MULTI-QUERY RETRIEVAL ==========",
        flush=True,
    )

    print("\nRunning original query search...", flush=True)
    original = vectorstore.similarity_search(question, k=k)

    print("Generating alternative queries...", flush=True)
    queries = generate_multiple_queries(question)

    if not queries:
        print("No alternative queries were generated.")
        return

    print("\nOriginal question:", question)
    print("\nAlternative queries:")

    for query in queries:
        print("-", query)

    rankings = []
    documents_by_id = {}

    all_queries = [question] + queries

    for i, query in enumerate(all_queries, start=1):
        print(
            f"\nSearching query {i}/{len(all_queries)}...",
            flush=True,
        )

        ids, found_documents = get_ranked_ids(
            vectorstore, query, k=20
        )

        rankings.append(ids)
        documents_by_id.update(found_documents)

    print("\nCombining search results with RRF...", flush=True)
    fused_ids = reciprocal_rank_fusion(rankings)

    final_results = [
        documents_by_id[source_id]
        for source_id in fused_ids
        if source_id in documents_by_id
    ][:k]

    print_results("BEFORE: Original Query Top 5", original)
    print_results("AFTER: Multi-Query + RRF Top 5", final_results)

    print(
        "\nExplanation: Multi-query retrieval searches the original "
        "question and three alternative phrasings. Reciprocal Rank "
        "Fusion combines their rankings to improve retrieval coverage."
    )


def main():
    vectorstore = load_vectorstore()

    print("Query Transformation Experiments")
    question = input("Enter a customer question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    run_query_rewriting(question, vectorstore)
    run_multi_query(question, vectorstore)


if __name__ == "__main__":
    main()
