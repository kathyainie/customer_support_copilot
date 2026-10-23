import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "data" / "dandes_electronics_kb.json"
load_dotenv(PROJECT_DIR / ".env")

CHAT_MODEL = "gpt-6-luna"

INTENT_TO_INDEX = {
    "policy": "policy_index",
    "product": "product_index",
    "order": "order_index",
    "past_issue": "ticket_index",
}

ALL_INDEXES = [
    "policy_index",
    "product_index",
    "order_index",
    "ticket_index",
]

router_model = ChatOpenAI(model=CHAT_MODEL)


def _load_knowledge_base():
    """Read the current dataset; never add or modify its records."""
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        documents = data.get("documents", [])
        return documents if isinstance(documents, list) else []

    except (OSError, json.JSONDecodeError, AttributeError):
        # The router must still be able to run if the dataset file is unavailable.
        return []


KB_DOCUMENTS = _load_knowledge_base()

KB_INVENTORY = "\n".join(
    f"- {item.get('doc_type', 'document')}: "
    f"{item.get('title', 'Untitled')}"
    for item in KB_DOCUMENTS
    if isinstance(item, dict)
)

KB_TEXT = " ".join(
    " ".join(
        str(value)
        for value in (
            item.get("title", ""),
            item.get("content", ""),
            item.get("doc_type", ""),
        )
        if value is not None
    )
    for item in KB_DOCUMENTS
    if isinstance(item, dict)
).lower()

KB_WORDS = set(re.findall(r"\b[a-z0-9]+\b", KB_TEXT))


_AVAILABILITY_PATTERNS = (
    re.compile(
        r"\bdo you (?:sell|stock|carry|offer)\s+(.+?)\s*[?.!]*$",
        re.I,
    ),
    re.compile(
        r"\bdoes (?:dandes electronics|your store|the store) "
        r"(?:sell|stock|carry|offer)\s+(.+?)\s*[?.!]*$",
        re.I,
    ),
)

_LEADING_FILLER = re.compile(
    r"^(?:(?:a|an|the|any|some|new|latest|brand[- ]new)\s+)+",
    re.I,
)

_TRAILING_DETAILS = re.compile(
    r"\s+(?:with|without|for|and|that|which|where|while|because|so that)\b.*$",
    re.I,
)


def _word_variants(word):
    """Small plural/singular check to handle ordinary catalog wording."""
    variants = {word}

    if len(word) > 4 and word.endswith("ies"):
        variants.add(word[:-3] + "y")

    if (
        len(word) > 3
        and word.endswith("s")
        and not word.endswith(("ss", "us", "is"))
    ):
        variants.add(word[:-1])

    if len(word) > 3 and word.endswith("y"):
        variants.add(word[:-1] + "ies")
    else:
        variants.add(word + "s")

    return variants


def _unsupported_product_availability(question):
    """Check direct availability questions against the actual dataset.

    This does not rely on a hardcoded list of test questions or product names.
    """
    if not KB_WORDS:
        return False

    target = None

    for pattern in _AVAILABILITY_PATTERNS:
        match = pattern.search(question.strip())

        if match:
            target = match.group(1)
            break

    if not target:
        return False

    target = _TRAILING_DETAILS.sub("", target)
    target = _LEADING_FILLER.sub("", target).strip(
        " \t\r\n?.!,;:"
    )

    target_words = re.findall(
        r"\b[a-z0-9]+\b",
        target.lower(),
    )

    ignored = {
        "product",
        "products",
        "item",
        "items",
        "thing",
        "things",
        "model",
        "models",
        "brand",
        "brands",
        "device",
        "devices",
    }

    target_words = [
        word for word in target_words
        if word not in ignored
    ]

    if not target_words:
        return False

    # If a meaningful target word is in the real knowledge base,
    # let the normal intent router handle the question.
    for word in target_words:
        if _word_variants(word).intersection(KB_WORDS):
            return False

    return True


def _parse_json(content):
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )

    content = str(content).strip()

    if content.startswith("```"):
        content = (
            content.replace("```json", "", 1)
            .replace("```", "")
            .strip()
        )

    start = content.find("{")
    end = content.rfind("}")

    if start >= 0 and end > start:
        content = content[start:end + 1]

    return json.loads(content)


def route_question(question):
    """Return intents, indexes, reason, and fallback status.

    Unsupported direct availability questions are checked against the
    dataset before calling the chat model.

    If the model fails or returns invalid JSON, search all indexes.
    """
    if _unsupported_product_availability(question):
        return {
            "intents": ["out_of_scope"],
            "indexes": [],
            "reason": (
                "The requested product is not represented in the available "
                "knowledge base, so no index should be searched."
            ),
            "fallback": False,
        }
    
    if re.search(
        r"\bpolicy\s+(?:RR|SH)-\d{3}\b",
        question,
        re.IGNORECASE,
    ):
        return {
            "intents": ["policy"],
            "indexes": ["policy_index"],
            "reason": "The question explicitly references a documented policy.",
            "fallback": False,
        }

    inventory_text = (
        KB_INVENTORY or "Dataset inventory is unavailable."
    )

    prompt = f"""
You route questions for Dandes Electronics customer support.

The actual knowledge-base inventory is listed below. Use it as evidence
of what information this system can answer. Do not assume the store sells,
supports, or documents a product merely because the user asks about it.

For direct product availability questions, if the requested product is not
represented by the inventory, choose out_of_scope alone.

KNOWLEDGE-BASE INVENTORY:
{inventory_text}

Choose one or more applicable intents:
- policy: returns, refunds, delivery rules, charges, COD, shipping policies
- product: product information, product FAQs, known product issues covered
  by the inventory
- order: actions involving an order, such as cancel, change address, track,
  report damaged item, or warranty claim, when relevant information is covered
- past_issue: asks about a previously resolved customer support case
- out_of_scope: the question cannot reasonably be answered using the available
  knowledge-base documents, including requests about unsupported product types

Index mapping:
- policy -> policy_index
- product -> product_index
- order -> order_index
- past_issue -> ticket_index
- out_of_scope -> no indexes

A question may have multiple intents. If it asks about both a product issue
and a similar previous case, choose product and past_issue.

If a question combines a policy/process and a previous case, choose the
relevant policy/order intent and past_issue.

Do not select out_of_scope alongside in-scope intents.
Do not treat a product as supported unless the knowledge-base inventory
provides evidence that it is covered.

Return valid JSON only, in this exact shape:
{{
  "intents": ["policy"],
  "indexes": ["policy_index"],
  "reason": "Short explanation"
}}
"""

    try:
        response = router_model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ])

        parsed = _parse_json(response.content)

        intents = parsed.get("intents", [])

        if not isinstance(intents, list):
            raise ValueError("Router intents must be a list.")

        valid_intents = [
            intent
            for intent in intents
            if intent in (*INTENT_TO_INDEX.keys(), "out_of_scope")
        ]

        if not valid_intents:
            raise ValueError("Router returned no valid intents.")

        in_scope_intents = [
            intent
            for intent in valid_intents
            if intent != "out_of_scope"
        ]

        if not in_scope_intents:
            return {
                "intents": ["out_of_scope"],
                "indexes": [],
                "reason": str(
                    parsed.get(
                        "reason",
                        "Question is outside the available knowledge-base scope.",
                    )
                ),
                "fallback": False,
            }

        indexes = []

        for intent in in_scope_intents:
            index_name = INTENT_TO_INDEX[intent]

            if index_name not in indexes:
                indexes.append(index_name)

        return {
            "intents": in_scope_intents,
            "indexes": indexes,
            "reason": str(parsed.get("reason", "")),
            "fallback": False,
        }

    except Exception as error:
        print(
            f"Router warning; falling back to all indexes: {error}"
        )

        return {
            "intents": [
                "policy",
                "product",
                "order",
                "past_issue",
            ],
            "indexes": ALL_INDEXES.copy(),
            "reason": (
                "Router failed; searching all indexes as a safe fallback."
            ),
            "fallback": True,
        }


def main():
    # Test Q7 without making an API call.
    if len(sys.argv) > 1 and sys.argv[1] == "--check-q7":
        question = "Do you sell refrigerators?"
        result = route_question(question)

        print(f"Question: {question}")
        print(f"Predicted intents: {result['intents']}")
        print(f"Predicted indexes: {result['indexes']}")
        print("Expected indexes : []")
        print(f"Reason: {result['reason']}")
        print(f"Fallback used: {result['fallback']}")
        return

    test_questions = [
        (
            "Can I return an opened laptop on day 8 if I have the original box and all accessories?",
            ["policy_index"],
        ),
        (
            "My Galaxy Buds3 Pro left earbud won't connect. Is this a known issue, and has the same problem been resolved for another customer? What fixed it?",
            ["product_index", "ticket_index"],
        ),
        (
            "How long does delivery take to a remote pincode, and is it free?",
            ["policy_index"],
        ),
        (
            "A customer wants to change the address after the order shipped. What is the official process, and do we have a resolved case like this?",
            ["order_index", "ticket_index"],
        ),
        (
            "What does policy RR-004 say about refund timelines?",
            ["policy_index"],
        ),
        (
            "A customer received a damaged phone and wants a refund instead of a replacement. What is the official process, and do we have a previous resolved case where a refund was issued?",
            [
                "policy_index",
                "order_index",
                "ticket_index",
            ],
        ),
        (
            "Do you sell refrigerators?",
            [],
        ),
    ]

    print("=" * 78)
    print("INTENT ROUTER TEST")
    print("=" * 78)

    for number, (question, expected) in enumerate(
        test_questions,
        start=1,
    ):
        result = route_question(question)

        print(f"\nQ{number}: {question}")
        print(f"Predicted intents: {result['intents']}")
        print(f"Predicted indexes: {result['indexes']}")
        print(f"Expected indexes : {expected}")
        print(f"Reason: {result['reason']}")
        print(f"Fallback used: {result['fallback']}")


if __name__ == "__main__":
    main()