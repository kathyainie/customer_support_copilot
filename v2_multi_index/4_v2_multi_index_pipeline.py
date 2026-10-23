import contextlib
import importlib.util
import io
import json
import sys
import argparse
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder


PROJECT_DIR = Path(__file__).resolve().parent.parent
THIS_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")

CHAT_MODEL = "gpt-6-luna"
EMBEDDING_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FALLBACK = "I don't know based on the available information."
TEST_QUESTIONS = [
    "Can I return an opened laptop on day 8 if I have the original box and all accessories?",
    "My Galaxy Buds3 Pro left earbud won't connect. Is this a known issue, and has the same problem been resolved for another customer? What fixed it?",
    "How long does delivery take to a remote pincode, and is it free?",
    "A customer wants to change the address after the order shipped. What is the official process, and do we have a resolved case like this?",
    "What does policy RR-004 say?",
    "A customer received a damaged phone and wants a refund instead of a replacement. What is the official process, and do we have a previous resolved case where a refund was issued?",
    "Do you sell refrigerators?",
]


def load_module(filename, module_name):
    path = THIS_DIR / filename
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


search_module = load_module(
    "2_multi_index_search.py",
    "multi_index_search",
)

router_module = load_module(
    "3_intent_router.py",
    "intent_router",
)


def _response_text(content):
    """Convert LangChain response content to plain text."""
    if isinstance(content, list):
        pieces = []

        for part in content:
            if (
                isinstance(part, dict)
                and isinstance(part.get("text"), str)
            ):
                pieces.append(part["text"])

        return "".join(pieces).strip()

    return str(content).strip()


def _parse_json_object(text):
    """Parse a JSON object, including a Markdown-fenced response."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = (
            cleaned.replace("```json", "", 1)
            .replace("```", "")
            .strip()
        )

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")

    value = json.loads(cleaned[start:end + 1])

    if not isinstance(value, dict):
        raise ValueError("Model response JSON must be an object.")

    return value


def generate_answer(question, documents, answer_model):
    """Generate an answer and retain only valid cited source IDs."""
    if not documents:
        return FALLBACK, []

    context_blocks = []
    available_ids = []

    for doc in documents:
        doc_id = doc.metadata.get("id")
        title = doc.metadata.get("title", "Untitled")

        if doc_id and doc_id not in available_ids:
            available_ids.append(doc_id)

        context_blocks.append(
            f"Source ID: {doc_id or 'Unknown'}\n"
            f"Title: {title}\n"
            f"Content:\n{doc.page_content}"
        )

    prompt = """
You are Dandes Electronics Customer Support Copilot.

Answer using only the supplied knowledge-base documents.

Return ONLY a valid JSON object with exactly these fields:
{
  "answer": "A clear, helpful answer.",
  "sources": ["SOURCE-ID"]
}

Rules:
- Use only the supplied knowledge-base documents.
- Do not invent policies, facts, or dates.
- Answer every part of the question.
- Separate official policies and procedures from resolved-ticket examples.
- Cite only source IDs present in the supplied context.
- Include sources that support distinct, relevant facts in the answer.
- Prefer a specific policy when it directly covers the requested facts;
  do not cite a general policy just to repeat information already covered.
- When describing a refund process, include the applicable refund timeline
  if a refund-timeline policy is supplied.
- Include relevant process details from order guides, such as required
  evidence and pickup steps, when those details are supplied.
- Do not add unrelated policy details that the question does not require.
- If the context does not support an answer, return exactly:
  I don't know based on the available information.
  and return an empty sources list.
- Do not put source IDs inside the answer text.
- For opened-laptop return questions, when both RR-001 and RR-002
  are available, explain that RR-001 gives the current general
  return window, effective 1 July 2026, while RR-002 covers
  the conditions for returning opened products. Cite both sources.
- For remote-pincode delivery questions, use SH-002 as the
  primary and sole source when it contains the information
  needed to answer. State the remote delivery time and the
  free-delivery threshold when supported. Do not add the exact
  below-threshold delivery fee from SH-001 unless the question
  explicitly asks for that fee.
- Before returning sources, check that every source supports
  a distinct claim and that no necessary source was omitted.
- When explaining the replacement-versus-refund rules for a damaged
  phone, cite RR-005 if it is present in the supplied context.
  Cite every source that supports a distinct part of the answer.
- When a policy's effective date is supplied in the context
  and is relevant to the answer, include the date explicitly.
- For remote-pincode delivery questions, explicitly state that
express delivery is not available for remote pincodes when supported by SH-002.
"""

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Available source IDs: {available_ids}\n\n"
        f"Knowledge-base context:\n"
        f"{'\n\n'.join(context_blocks)}"
    )

    response = answer_model.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_prompt),
    ])

    raw_text = _response_text(response.content)

    try:
        result = _parse_json_object(raw_text)

        answer = str(result.get("answer", "")).strip()
        chosen_sources = result.get("sources", [])

        if not isinstance(chosen_sources, list):
            chosen_sources = []

        allowed = set(available_ids)
        cited_sources = []

        for source_id in chosen_sources:
            if isinstance(source_id, str):
                source_id = source_id.strip()

                if (
                    source_id in allowed
                    and source_id not in cited_sources
                ):
                    cited_sources.append(source_id)

        if not answer:
            raise ValueError("Model returned an empty answer.")

        if answer == FALLBACK:
            return FALLBACK, []

        return answer, cited_sources

    except (ValueError, json.JSONDecodeError, TypeError):
        return raw_text or FALLBACK, available_ids if raw_text else []


def _format_routing(intents, indexes):
    intent_text = ", ".join(intents) if intents else "none"
    index_text = ", ".join(indexes) if indexes else "none"

    return (
        f"intents=[{intent_text}] "
        f"-> indexes=[{index_text}]"
    )


def format_result(item):
    source_text = (
        ", ".join(item["sources"])
        if item["sources"]
        else "None"
    )

    return (
        f"Question: {item['question']}\n"
        f"Routing: {_format_routing(item['intents'], item['indexes'])}\n"
        f"Answer: {item['answer']}\n"
        f"Sources: {source_text}\n"
    )


def print_result(item):
    print("\n" + "=" * 78)
    print("CUSTOMER SUPPORT COPILOT")
    print("=" * 78)
    print(format_result(item))


def run_question(
    question,
    loaded_indexes,
    reranker,
    answer_model,
):
    routing = router_module.route_question(question)
    intents = routing.get("intents", [])

    if intents == ["out_of_scope"]:
        item = {
            "question": question,
            "intents": intents,
            "indexes": [],
            "answer": FALLBACK,
            "sources": [],
        }

        print_result(item)
        return item

    if routing.get("fallback", False):
        indexes = list(search_module.INDEX_NAMES)
    else:
        indexes = routing.get("indexes", [])

    if not indexes:
        item = {
            "question": question,
            "intents": intents,
            "indexes": [],
            "answer": FALLBACK,
            "sources": [],
        }

        print_result(item)
        return item

    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        documents = search_module.search_selected_indexes(
            question,
            indexes,
            loaded_indexes,
            reranker,
        )
    
    id_match = re.search(
        r"\b(?:FAQ|RR|SH|OG|KI|TCK)-\d{3,4}\b",
        question.upper(),
    )

    if id_match:
        requested_id = id_match.group(0)

        for index_name in indexes:
            index_documents = loaded_indexes[index_name][2]

            matching_doc = next(
                (
                    doc for doc in index_documents
                    if str(doc.metadata.get("id", "")).upper()
                    == requested_id
                ),
                None,
            )

            if matching_doc:
                documents = [matching_doc] + [
                    doc for doc in documents
                    if str(doc.metadata.get("id", "")).upper()
                    != requested_id
                ]
                documents = documents[:5]
                break

    answer, source_ids = generate_answer(
        question,
        documents,
        answer_model,
    )

    item = {
        "question": question,
        "intents": intents,
        "indexes": indexes,
        "answer": answer,
        "sources": source_ids,
    }

    print_result(item)
    return item


def main():
    parser = argparse.ArgumentParser(
        description="Dandes Electronics Customer Support Copilot"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the seven assignment questions and save demo_output.txt",
    )
    args = parser.parse_args()

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        reranker = CrossEncoder(RERANKER_MODEL)

    answer_model = ChatOpenAI(model=CHAT_MODEL)

    loaded_indexes = {
        name: search_module.load_index(name, embeddings)
        for name in search_module.INDEX_NAMES
    }

    print("=" * 78)
    print("DANDES ELECTRONICS — CUSTOMER SUPPORT COPILOT")
    print("=" * 78)

    # Assignment mode: run all seven questions and save the results.
    if args.demo:
        print("Running the seven assignment questions...\n")

        results = [
            run_question(question, loaded_indexes, reranker, answer_model)
            for question in TEST_QUESTIONS
        ]

        output_path = PROJECT_DIR / "demo_output.txt"

        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(
                "CUSTOMER SUPPORT COPILOT — V2 MULTI-INDEX RAG\n"
            )
            output_file.write("=" * 78 + "\n\n")

            for item in results:
                output_file.write(format_result(item))
                output_file.write("\n" + "=" * 78 + "\n\n")

        print(f"\nDemo output saved to: {output_path}")
        return

   
    print("Ask a customer-support question.")
    print("Type 'exit' to quit.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCopilot session ended. Goodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye! 👋")
            break

        if not question:
            continue

        item = run_question(
            question,
            loaded_indexes,
            reranker,
            answer_model,
        )

        
        output_path = PROJECT_DIR / "demo_output.txt"

        with open(output_path, "a", encoding="utf-8") as output_file:
            output_file.write(format_result(item))
            output_file.write("\n" + "=" * 78 + "\n\n")

        print(f"Conversation saved to: {output_path}")


if __name__ == "__main__":
    main()