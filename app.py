from __future__ import annotations

import streamlit as st

from clients import (
    build_context,
    detect_device_source,
    make_openai_client,
    make_search_client,
    retrieve,
)
from config import get_settings
from rag import build_messages


st.set_page_config(page_title="Vet-eye CareBot", page_icon="🩺", layout="centered")


@st.cache_resource(show_spinner=False)
def _clients():
    """Created once per server process and reused across reruns/sessions."""
    s = get_settings()
    return s, make_openai_client(s), make_search_client(s)


def main() -> None:
    try:
        s, openai_client, search_client = _clients()
    except Exception as e:
        st.error(
            "Configuration error. Copy `.env.example` to `.env` and fill in your "
            f"Azure credentials.\n\nDetails: {e}"
        )
        st.stop()

    st.title("Vet-eye CareBot")
    st.caption("Asystent wsparcia technicznego L1 · L1 technical support assistant")

    with st.sidebar:
        show_sources = st.toggle("Show retrieved sources", value=True)
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.device = None
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources") and show_sources:
                _render_sources(msg["sources"])

    prompt = st.chat_input("Zadaj pytanie o urządzenie… / Ask about your device…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Capture the device from the user's message text. If this turn doesn't name
    # one, fall back to the device named earlier in the conversation so follow-up
    # questions ("a teraz podkręć kontrast") still filter to the right manual.
    device = detect_device_source(prompt) or st.session_state.get("device")
    st.session_state.device = device

    with st.chat_message("assistant"):
        # 1) Retrieve grounding context (hybrid + semantic), pinned to the device.
        with st.spinner("Searching manuals…"):
            docs = retrieve(search_client, openai_client, s, prompt, device_source=device)
        context = build_context(docs)

        # 2) Build the message list: system + recent history + grounded turn.
        #    Exclude the just-added user message from history.
        messages = build_messages(prompt, context, st.session_state.messages[:-1])

        # 3) Stream the answer.
        answer = st.write_stream(_stream_answer(openai_client, s, messages))

        if show_sources:
            _render_sources(docs)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": docs}
    )


def _stream_answer(openai_client, s, messages):
    """Yield answer tokens as they arrive from gpt-5-mini."""
    stream = openai_client.chat.completions.create(
        model=s.chat_deployment,
        messages=messages,
        stream=True,
        max_completion_tokens=s.max_completion_tokens,
        reasoning_effort=s.reasoning_effort,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def _render_sources(docs: list[dict]) -> None:
    if not docs:
        return
    with st.expander(f"Sources ({len(docs)})"):
        for i, d in enumerate(docs, start=1):
            page = f" · p.{d['page']}" if d.get("page") is not None else ""
            score = d.get("score")
            score_str = f" · score {score:.2f}" if isinstance(score, (int, float)) else ""
            st.markdown(f"**[{i}] {d['source']}**{page}{score_str}")
            st.caption(d["content"][:300] + ("…" if len(d["content"]) > 300 else ""))


if __name__ == "__main__":
    main()
