import json
import re
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "data" / "dandes_electronics_kb.json"
VECTORSTORE_DIR = PROJECT_DIR / "chroma_db"

load_dotenv(PROJECT_DIR / ".env")

COLLECTION_NAME = "dandes_v1"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-6-luna"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FALLBACK = "I don't know based on the available information."

VALID_DOC_TYPES = {
    "product_faq",
    "known_issue",
    "return_policy",
    "shipping_policy",
    "order_guide",
    "resolved_ticket",
}


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)["documents"]


def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )



def transform_query(question):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Rewrite the customer's question into three alternative
search queries for a customer support knowledge base.
Preserve product names, policy codes, and order details.
Return exactly three queries, one per line, without numbering."""
        ),
        ("human", "{question}"),
    ])

    model = ChatOpenAI(model=CHAT_MODEL)
    response = (prompt | model).invoke({"question": question})

    queries = [
        re.sub(r"^\s*\d+[\.\)\-]\s*", "", line).strip()
        for line in response.content.splitlines()
        if line.strip()
    ]

    queries = [query for query in queries if query]

    return queries[:3]

def reciprocal_rank_fusion(rankings, rrf_k=60):
    scores = {}

    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] = scores.get(document_id, 0) + (
                1 / (rrf_k + rank)
            )

    return sorted(scores, key=scores.get, reverse=True)


def hybrid_candidates(question, vectorstore, raw_documents,
                      doc_type=None, candidate_k=20):
    search_filter = {"doc_type": doc_type} if doc_type else None

    semantic_results = vectorstore.similarity_search(
        question,
        k=candidate_k,
        filter=search_filter,
    )

    semantic_ids = [
        doc.metadata["id"] for doc in semantic_results
    ]

    filtered_raw = [
        item for item in raw_documents
        if doc_type is None or item["doc_type"] == doc_type
    ]

    tokenized_corpus = [
        f"{item['title']} {item['content']}".lower().split()
        for item in filtered_raw
    ]

    if not tokenized_corpus:
        return []

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(question.lower().split())

    ranked_indices = sorted(
        range(len(filtered_raw)),
        key=lambda i: scores[i],
        reverse=True,
    )[:candidate_k]

    bm25_ids = [
        filtered_raw[i]["id"] for i in ranked_indices
    ]

    return reciprocal_rank_fusion(
        [semantic_ids, bm25_ids]
    )[:candidate_k]

def advanced_search(question, doc_type=None, candidate_k=20, top_k=5):
    raw_documents = load_dataset()
    documents_by_id = {item["id"]: item for item in raw_documents}
    vectorstore = load_vectorstore()

    referenced_ids = re.findall(
        r"\b[A-Z]{2,5}-\d{3,4}\b",
        question.upper(),
    )

    queries = [question] + transform_query(question)
    all_rankings = []

    for query in queries:
        ranking = hybrid_candidates(
            query,
            vectorstore,
            raw_documents,
            doc_type=doc_type,
            candidate_k=candidate_k,
        )
        all_rankings.append(ranking)

    fused_ids = reciprocal_rank_fusion(all_rankings)

    candidate_ids = referenced_ids + fused_ids
    candidates = []
    seen_ids = set()

    for doc_id in candidate_ids:
        if doc_id in seen_ids or doc_id not in documents_by_id:
            continue

        document = documents_by_id[doc_id]

        if doc_type is not None and document["doc_type"] != doc_type:
            continue

        candidates.append(document)
        seen_ids.add(doc_id)

        if len(candidates) >= candidate_k:
            break

    if not candidates:
        return []

    reranker = CrossEncoder(RERANKER_MODEL)
    pairs = [
        (question, f"{item['title']}\n{item['content']}")
        for item in candidates
    ]
    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    referenced_in_candidates = [
        item for item in candidates
        if item["id"] in referenced_ids
    ]

    other_ranked_documents = [
        item for item, score in reranked
        if item["id"] not in referenced_ids
    ]

    final_results = (
        referenced_in_candidates + other_ranked_documents
    )[:top_k]

    print("\nAdvanced RAG retrieved sources:")
    for item in final_results:
        print(f"- {item['id']} ({item['doc_type']}): {item['title']}")

    return final_results

def generate_answer(question, documents, chat_history=None):
    if not documents:
        return FALLBACK

    context = "\n\n---\n\n".join(
        f"Source ID: {item['id']}\n"
        f"Title: {item['title']}\n"
        f"Content: {item['content']}"
        for item in documents
    )

    prompt = f"""
You are a customer support assistant for Dandes Electronics.
Answer using only the supplied context. Do not invent facts.
If the context does not contain enough information to answer,
reply with exactly:
{FALLBACK}

For every factual answer, include the source IDs supporting it.
Put the source IDs on a separate line starting with Sources:.
If the supplied context does not support the answer, use the exact fallback.
Never invent a source ID.

Previous conversation:
{chr(10).join(chat_history) if chat_history else "None"}

Context:
{context}

Customer question:
{question}
"""

    model = ChatOpenAI(model=CHAT_MODEL)
    response = model.invoke(prompt)

    return response.content


def run_pipeline():
    chat_history = []

    print("Dandes Electronics — Advanced RAG")
    print("Optional document types: product_faq, known_issue,")
    print("return_policy, shipping_policy, order_guide, resolved_ticket")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Customer question: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        doc_type = input(
            "Optional document-type filter "
            "(product_faq, known_issue, return_policy, "
            "shipping_policy, order_guide, resolved_ticket; "
            "press Enter to skip): "
        ).strip() or None

        if doc_type and doc_type not in VALID_DOC_TYPES:
            print("Invalid document type. Please try again.\n")
            continue

        documents = advanced_search(question, doc_type=doc_type)
        answer = generate_answer(question, documents, chat_history)

        print(f"\nAssistant: {answer}\n")

        chat_history.append(f"Customer: {question}")
        chat_history.append(f"Assistant: {answer}")
        chat_history = chat_history[-6:]


if __name__ == "__main__":
    run_pipeline()
