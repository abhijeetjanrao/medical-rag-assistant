import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Medical RAG Assistant", page_icon=None)
st.title("Medical RAG Assistant")
st.caption("Answers are grounded in WHO/CDC guidelines and cited medical references. Not a substitute for professional medical advice.")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn.get("sources"):
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for s in turn["sources"]:
                    st.markdown(
                        f"**{s['document_name']}**, page {s['page_number']} "
                        f"(relevance: {s['relevance_score']:.2f})\n\n"
                        f"> {s['excerpt']}"
                    )

query = st.chat_input("Ask a medical question...")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"query": query, "conversation_id": st.session_state.conversation_id},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state.conversation_id = data["conversation_id"]

                st.write(data["answer"])
                if data["guardrail_flagged"]:
                    st.info("This response was flagged by the guardrail system for careful handling.")
                if data["sources"]:
                    with st.expander(f"Sources ({len(data['sources'])})"):
                        for s in data["sources"]:
                            st.markdown(
                                f"**{s['document_name']}**, page {s['page_number']} "
                                f"(relevance: {s['relevance_score']:.2f})\n\n"
                                f"> {s['excerpt']}"
                            )
                st.caption(f"Latency: {data['latency_ms']:.0f} ms")

                st.session_state.history.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data["sources"],
                })
            except requests.exceptions.RequestException as e:
                st.error(f"Couldn't reach the backend: {e}")
