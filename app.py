import os
import sys
from pathlib import Path

# Must be set before protobuf is imported (streamlit/chromadb).
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

import logging
import time
import warnings

import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(level=logging.WARNING, format="%(message)s")
for _logger in (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "chromadb",
    "langchain",
    "langchain_community",
    "streamlit",
):
    logging.getLogger(_logger).setLevel(logging.ERROR)

log = logging.getLogger("hr_app")
log.setLevel(logging.INFO)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config.config import GROQ_API_KEY, MODEL_CONFIG, UI_CONFIG
from src.core.document_processor import load_vector_store, run_document_processing
from src.core.groq_client import GroqClient
from src.core.rag_chain import create_qa_chain
from utils.helpers import get_user_intent, log_feedback

HR_DOCS_DIR = "./hr_documents"
VECTOR_STORE_DIR = "./hr_chroma_db"


def format_chat_history(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def handle_casual_chat(groq_client: GroqClient, question: str, chat_history: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly HR assistant. Respond briefly and naturally to casual "
                "conversation. If the user asks HR policy questions, suggest they ask about "
                "company policies."
            ),
        },
        {
            "role": "user",
            "content": f"Conversation so far:\n{chat_history}\n\nUser: {question}",
        },
    ]
    response = groq_client.chat_completion(messages=messages, max_tokens=300)
    return response["choices"][0]["message"]["content"]


@st.cache_resource(show_spinner="Loading knowledge base and models...")
def init_app():
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to .env locally or Streamlit Secrets."
        )

    vector_store = load_vector_store(VECTOR_STORE_DIR)
    if vector_store is None:
        log.info("Building knowledge base from HR documents...")
        vector_store = run_document_processing(HR_DOCS_DIR)
    if vector_store is None:
        raise RuntimeError("Failed to load or build the vector store.")

    groq_client = GroqClient(api_key=GROQ_API_KEY)
    process_query = create_qa_chain(
        vector_store,
        groq_client,
        model_name=MODEL_CONFIG["main_model"],
    )
    log.info("Ready")
    return groq_client, process_query


def main():
    st.set_page_config(
        page_title=UI_CONFIG["page_title"],
        page_icon=UI_CONFIG["page_icon"],
        layout=UI_CONFIG["layout"],
    )

    st.title("Mati Carbon HR AI Assistant")
    st.caption("Ask questions about company HR policies and benefits.")

    try:
        groq_client, process_query = init_app()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Settings")
        st.write(f"**LLM:** `{MODEL_CONFIG['main_model']}`")
        st.write(f"**Embeddings:** `{MODEL_CONFIG['embedding_model']}`")

        if st.button("Rebuild knowledge base", type="secondary"):
            with st.spinner("Rebuilding index from HR documents..."):
                init_app.clear()
                run_document_processing(HR_DOCS_DIR)
                st.success("Knowledge base rebuilt. Reloading...")
                st.rerun()

        st.divider()
        st.markdown("Place HR PDFs in `backend/hr_documents/`.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption("Sources: " + ", ".join(message["sources"]))

    if prompt := st.chat_input("Ask an HR policy question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    start = time.time()
                    chat_history = format_chat_history(st.session_state.messages[:-1])
                    intent = get_user_intent(prompt)

                    if intent == "MALICIOUS_INPUT":
                        answer = (
                            "I can't help with that request. Please ask a question about "
                            "HR policies or benefits."
                        )
                        sources = []
                    elif intent == "CASUAL_CONVERSATION":
                        answer = handle_casual_chat(groq_client, prompt, chat_history)
                        sources = []
                    else:
                        answer, sources = process_query(
                            {
                                "question": prompt,
                                "chat_history": chat_history,
                            }
                        )

                    st.markdown(answer)
                    if sources:
                        st.caption("Sources: " + ", ".join(sources))

                    elapsed = time.time() - start
                    src = ", ".join(sources) if sources else "-"
                    log.info(f"OK | {intent} | {elapsed:.1f}s | {src}")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Helpful", key=f"helpful_{len(st.session_state.messages)}"):
                            log_feedback(prompt, answer, "helpful")
                            st.toast("Thanks for your feedback!")
                    with col2:
                        if st.button("Not helpful", key=f"not_helpful_{len(st.session_state.messages)}"):
                            log_feedback(prompt, answer, "not_helpful")
                            st.toast("Feedback recorded.")

                except Exception as exc:
                    log.error(f"FAILED | {exc}")
                    st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":
    main()
