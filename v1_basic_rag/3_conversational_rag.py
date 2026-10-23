from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = PROJECT_DIR / "chroma_db"

load_dotenv(PROJECT_DIR / ".env")

COLLECTION_NAME = "dandes_v1"
EMBEDDING_MODEL = "text-embedding-3-small"


CHAT_MODEL = "gpt-6-luna"

FALLBACK = "I don't know based on the available information."


def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )


def retrieve_documents(question, k=5):
    vectorstore = load_vectorstore()

    return vectorstore.similarity_search(question, k=k)


def generate_answer(question, documents, chat_history):
    context_parts = []

    for document in documents:
        source_id = document.metadata.get("id", "Unknown")
        context_parts.append(
            f"Source ID: {source_id}\n{document.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a customer support assistant for Dandes Electronics.

Answer the customer's question using only the supplied context.
Do not invent facts or use outside knowledge.
If the context does not contain enough information to answer,
reply with exactly:
{FALLBACK}

Include the source IDs used to support your answer.
Put the source IDs on a new line starting with Sources:

Context:
{context}

Customer question:
{question}
"""

    chat_model = ChatOpenAI(model=CHAT_MODEL)

    messages = [
        *chat_history,
        HumanMessage(content=prompt),
    ]

    response = chat_model.invoke(messages)
    return response.content


def conversational_rag():
    chat_history = []

    print("Dandes Electronics Customer Support Copilot")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        documents = retrieve_documents(question)
        answer = generate_answer(question, documents, chat_history)

        print(f"\nAssistant: {answer}\n")

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))


if __name__ == "__main__":
    conversational_rag()
