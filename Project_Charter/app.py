import streamlit as st
from rag_qa import answer  # reuse logic

st.set_page_config(page_title="Charter Communication Sample RAG Assistant", page_icon="🛠️")
st.title(" Charter Communication RAG Assistant")

# Examples
st.markdown("#### Example Questions:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button(" what does SYS_ERR_DEVICE_TIMEOUT mean?"):
        st.session_state.query = " what does SYS_ERR_DEVICE_TIMEOUT mean?"
with col2:
    if st.button("Why did my firewall drop invalid packets?"):
        st.session_state.query = "Why did my firewall drop invalid packets?"
with col3:
    if st.button("What does INTERFACE_DOWN mean?"):
        st.session_state.query = "why does INTERFACE_DOWN mean?"

# Input box with a placeholder and optional state override
query = st.text_input("Ask your network question…", st.session_state.get("query", ""))

if query:
    with st.spinner("Thinking…"):
        resp = answer(query)
    st.markdown("**Answer:**")
    st.write(resp)
