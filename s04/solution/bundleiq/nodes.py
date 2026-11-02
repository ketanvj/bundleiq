"""
bundleiq/nodes.py
-----------------
Node functions for the BundleIQ graph.

Session 4 adds ChromaDB RAG:
  retrieve_docs() -- new node: fetches relevant policy chunks from ChromaDB
  respond()       -- updated: builds context-augmented system prompt from retrieved chunks

classify(), route_query(), decline() carry forward from Session 3,
with one change: classify() now returns IN_SCOPE / OUT_OF_SCOPE (2-way) instead
of SIMPLE / COMPLEX / OUT_OF_SCOPE (3-way). The respond() node handles escalation
internally when no docs are retrieved.
"""
from langchain_chroma import Chroma

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    CLASSIFY_SYSTEM, DECLINE_RESPONSE, ESCALATE_RESPONSE,
    EMBED_MODEL, RETRIEVAL_K, RETRIEVAL_SCORE_THRESHOLD, SYSTEM_PROMPT, VECTORSTORE_DIR,
)
from .state import BundleIQState
from .tools import classifier_llm, llm

vectorstore = None  # shared across calls; initialised once by _init_vectorstore()

BLOCKLIST = [
    "ignore all previous",
    "forget everything",
    "you are now",
    "disregard your system",
    "act as",
    "jailbreak",
]


def _init_vectorstore() -> None:
    global vectorstore
    if vectorstore is not None:  # already loaded — skip the 90 MB model reload
        return
    try:
        embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
        )
        count = vectorstore._collection.count()
        if count == 0:
            print(f"[BundleIQ] Vectorstore opened but is EMPTY (0 chunks).")
            print(f"  Run 'python data/ingest.py' from the bundleiq/ root to populate it.")
        else:
            print(f"[BundleIQ] Vectorstore ready — {count} chunks loaded.")
    except Exception as e:
        print(f"[BundleIQ] Could not load vectorstore: {e}")
        print("  Run 'python data/ingest.py' to create it.")


def classify(state: BundleIQState) -> dict:
    msg = state["customer_message"].strip()

    if not msg or len(msg) < 3 or len(msg) > 500:
        return {"query_type": "OUT_OF_SCOPE", "retrieved_docs": []}

    if any(phrase in msg.lower() for phrase in BLOCKLIST):
        return {"query_type": "OUT_OF_SCOPE", "retrieved_docs": []}

    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=msg),
    ]
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()
        if query_type not in {"IN_SCOPE", "OUT_OF_SCOPE"}:
            query_type = "IN_SCOPE"
    except Exception as e:
        print(f"[BundleIQ] Classification error: {e}")
        query_type = "IN_SCOPE"
    # Clear any retrieved_docs from the previous turn so stale results never leak.
    return {"query_type": query_type, "retrieved_docs": []}


def retrieve_docs(state: BundleIQState) -> dict:
    _init_vectorstore()
    if vectorstore is None:
        return {"retrieved_docs": []}
    try:
        results   = vectorstore.similarity_search_with_relevance_scores(
            state["customer_message"], k=RETRIEVAL_K
        )
        if not results:
            print(f"[BundleIQ] No chunks returned — vectorstore may be empty. Run 'python data/ingest.py'.")
            return {"retrieved_docs": []}
        retrieved = []
        for doc, score in results:
            if score >= RETRIEVAL_SCORE_THRESHOLD:
                retrieved.append(
                    f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                )
            else:
                print(
                    f"[BundleIQ] Chunk skipped (score {score:.2f} < {RETRIEVAL_SCORE_THRESHOLD}): "
                    f"{doc.metadata.get('source', 'unknown')}"
                )
        print(f"[BundleIQ] Retrieval: {len(results)} candidates, {len(retrieved)} passed threshold ({RETRIEVAL_SCORE_THRESHOLD}).")
    except Exception as e:
        print(f"[BundleIQ] Retrieval error: {e}")
        retrieved = []
    return {"retrieved_docs": retrieved}


def respond(state: BundleIQState) -> dict:
    history   = state.get("history", [])
    retrieved = state.get("retrieved_docs", [])

    # If no docs were retrieved the knowledge base cannot support an answer.
    # Escalate directly without an LLM call — fast and deterministic.
    if not retrieved:
        new_history = history + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": ESCALATE_RESPONSE},
        ]
        return {"response": ESCALATE_RESPONSE, "history": new_history}

    context_block  = "\n\n".join(f"<policy_chunk>\n{chunk}\n</policy_chunk>" for chunk in retrieved)
    system_content = (
        SYSTEM_PROMPT
        + f"\n\n<retrieved_documents>\n{context_block}\n</retrieved_documents>"
    )

    messages = [SystemMessage(content=system_content)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result        = llm.invoke(messages)
        response_text = result.content
    except Exception as e:
        print(f"[BundleIQ] LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    new_history = history + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": response_text},
    ]
    return {"response": response_text, "history": new_history}


def escalate(state: BundleIQState) -> dict:
    """Kept for Option A fallback — not wired into the graph under Option B."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}


def decline(state: BundleIQState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}


def route_query(state: BundleIQState) -> str:
    qt = state.get("query_type", "IN_SCOPE")
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "retrieve_docs"
