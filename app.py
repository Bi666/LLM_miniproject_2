import streamlit as st
from agents import Head_Agent

st.set_page_config(page_title="ML Textbook Chatbot", page_icon="📚", layout="centered")
st.title("ML Textbook Multi-Agent Chatbot")
st.caption("Powered by GPT-4.1-nano + Pinecone RAG")

# --- Sidebar for API keys ---
with st.sidebar:
    st.header("Configuration")
    openai_key = st.text_input("OpenAI API Key", type="password")
    pinecone_key = st.text_input("Pinecone API Key", type="password")
    index_name = st.text_input("Pinecone Index Name", value="machine-learning-textbook")
    namespace = st.text_input("Namespace", value="ns2500")

    if st.button("Reset Conversation"):
        st.session_state.messages = []
        if "head_agent" in st.session_state:
            st.session_state.head_agent.reset_conversation()
        st.rerun()

# --- Initialize head agent ---
if openai_key and pinecone_key:
    if "head_agent" not in st.session_state:
        with st.spinner("Initializing agents..."):
            st.session_state.head_agent = Head_Agent(
                openai_key=openai_key,
                pinecone_key=pinecone_key,
                pinecone_index_name=index_name,
                namespace=namespace,
            )
        st.success("Agents ready!")
else:
    st.info("Please enter your OpenAI and Pinecone API keys in the sidebar to get started.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "agent_path" in message:
            st.caption(f"Agent path: {message['agent_path']}")

if prompt := st.chat_input("Ask me about Machine Learning..."):
    if "head_agent" not in st.session_state:
        st.warning("Please configure API keys first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, agent_path = st.session_state.head_agent.process_query(prompt)
            st.markdown(response)
            st.caption(f"Agent path: {agent_path}")

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "agent_path": agent_path}
        )
