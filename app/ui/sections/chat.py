"""Chat section — RAG-powered data Q&A + NL chart commands."""

import pandas as pd
import streamlit as st
from ..helpers import safe_df_for_display
from ...chat.rag import DataRAG
from ...chat.assistant import chat_with_llm
from ...viz.charts import parse_nl_chart, create_chart


def render_chat(df: pd.DataFrame, settings):
    """Render the chat section with RAG retrieval."""
    st.subheader("💬 Chat with your data")
    st.caption(f"Local AI: {settings.llm_model} | RAG-powered context retrieval")

    # Build RAG index (cached in session state)
    if "rag_index" not in st.session_state or st.session_state.get("rag_df_id") != id(df):
        with st.spinner("Building data index..."):
            st.session_state.rag_index = DataRAG(df)
            st.session_state.rag_df_id = id(df)
        st.success(f"✓ Indexed {st.session_state.rag_index.chunk_count} knowledge chunks")

    rag: DataRAG = st.session_state.rag_index

    # NL chart commands
    nl_cmd = st.text_input("📊 Quick chart command", placeholder="e.g. histogram salary, scatter age vs income, box price by category")
    if st.button("Render chart") and nl_cmd:
        spec = parse_nl_chart(nl_cmd, df)
        if spec is None:
            st.warning("Could not parse command. Try: `histogram <column>`, `scatter <x> vs <y>`, `box <column>`")
        else:
            fig = create_chart(spec, df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not create chart.")

    # LLM chat
    st.markdown("---")
    
    user_msg = st.text_input("🤔 Ask a question about your data")
    if st.button("Ask") and user_msg:
        with st.spinner(f"Thinking ({settings.llm_model})..."):
            reply = chat_with_llm(
                api_base=settings.llm_api_base,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                user_message=user_msg,
                rag=rag,
                history=st.session_state.get("chat_history"),
            )
        st.session_state.setdefault("chat_history", [])
        st.session_state["chat_history"].append({"role": "user", "content": user_msg})
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})

        # Show retrieved context for transparency
        with st.expander("🔍 Retrieved context chunks"):
            for chunk in rag.retrieve(user_msg, top_k=5):
                st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)

    # Chat history
    hist = st.session_state.get("chat_history", [])
    if hist:
        for m in hist[-8:]:
            prefix = "**You:** " if m["role"] == "user" else "**Assistant:** "
            st.markdown(prefix + m["content"])

    # Verify facts panel
    with st.expander("Verify chat claims against data", expanded=False):
        st.write({"rows": int(df.shape[0]), "columns": int(df.shape[1])})
        dtypes_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
        st.dataframe(safe_df_for_display(dtypes_df))
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["column", "missing"]
        st.dataframe(safe_df_for_display(missing_df))
