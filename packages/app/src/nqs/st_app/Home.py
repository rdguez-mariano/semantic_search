from typing import Any, Dict, List

import streamlit as st
from streamlit_js_eval import get_page_location

from nqs.llm.rag_retriever import DocMetadata
from nqs.st_app.utils import generate_answer, scrape_and_initialize_retriever

# App title
st.set_page_config(page_title="Semantic search")
st.title("Semantic search of articles")
st.markdown(
    """
Semantic search from feeds.
Sources:
* vsd.fr
* public.fr
"""
)

if "current_url" not in st.session_state.keys():
    _current_url = get_page_location()
    if _current_url is None:
        st.stop()
    st.session_state["current_url"] = _current_url["origin"]

current_url = st.session_state["current_url"]


def update_source_data():
    with st.spinner("scraping sources and recreating vector db..."):
        scrape_and_initialize_retriever()


def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Comment puis-je vous aider aujourd'hui ?",
        }
    ]


def display_message(msg: Dict[str, Any]):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "refs" in message:
            st.write("\n\n **Links**:\n")
            refs: List[DocMetadata] = message["refs"]
            for hid, doc in enumerate(refs):
                st.write(f"\n- [{doc.title}]({doc.url})")
                st.image(doc.thumb, width=200)
    st.write("")


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    clear_chat_history()


# Display or clear chat messages
for message in st.session_state.messages:
    display_message(message)


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)
st.sidebar.button("Update from sources", on_click=update_source_data)


if prompt := st.chat_input(placeholder="votre question", disabled=False):
    message = {"role": "user", "content": prompt}
    st.session_state.messages.append(message)
    display_message(message)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] == "user":
    with st.spinner("..."):
        response = generate_answer(str(prompt))
        if response.answer is not None:

            message = {
                "role": "assistant",
                "content": "**Anwser**: " + response.answer,
                "refs": response.source_docs,
            }
        elif response.example_questions is not None:
            full_response = (
                "Pas réponse... peut-être dû à la question, "
                "voici quelques examples:\n"
            )
            for item in response.example_questions:
                full_response += "\n* " + item
            message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
    display_message(message)
