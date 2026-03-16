import os
from typing import Dict, List, Optional, Tuple

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from pinecone import Pinecone


INDEX_NAME = "machine-learning-textbook"
NAMESPACE = "ns2500"
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_KEY = os.environ.get("PINECONE_API_KEY")


class Obnoxious_Agent:
    def __init__(self, client: OpenAI) -> None:
        self.client = client
        self.prompt = (
            "You are a content-moderation classifier. "
            "Determine whether the following user message is obnoxious, rude, "
            "offensive, or contains hate speech / personal attacks.\n"
            "Respond with ONLY 'Yes' if the message is obnoxious, or 'No' if it is not.\n"
            "Do not provide any explanation."
        )

    def check_query(self, query: str) -> bool:
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=5,
        )
        return response.choices[0].message.content.strip().lower().startswith("yes")


class Context_Rewriter_Agent:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client

    def rephrase(self, user_history: List[Dict[str, str]], latest_query: str) -> str:
        history_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in user_history[-8:]
        )
        system_msg = (
            "You are a query-rewriting assistant for a Machine Learning textbook chatbot. "
            "Given the conversation history and the user's latest message, rewrite the "
            "latest message into a FULLY self-contained ML question.\n\n"
            "Rules:\n"
            "- Replace pronouns with the specific ML concepts they refer to.\n"
            "- If already self-contained, return unchanged.\n"
            "- Return ONLY the rewritten query."
        )
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{history_str}\n\n"
                        f"Latest user message: {latest_query}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()


class Query_Agent:
    def __init__(
        self,
        pinecone_index,
        openai_client: OpenAI,
        embeddings: OpenAIEmbeddings,
        namespace: str = NAMESPACE,
    ) -> None:
        self.client = openai_client
        self.vectorstore = PineconeVectorStore(
            index=pinecone_index,
            embedding=embeddings,
            text_key="text",
            namespace=namespace,
        )
        self.prompt = (
            "You are a relevance classifier for a Machine Learning textbook assistant. "
            "Given the user query, determine if it is related to Machine Learning, "
            "AI, data science, statistics, or typical ML textbook topics. "
            "Respond with ONLY 'Yes' if relevant, or 'No' if not relevant."
        )

    def query_vector_store(self, query: str, k: int = 5) -> List:
        return self.vectorstore.similarity_search(query, k=k)

    def is_relevant_topic(self, query: str) -> bool:
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=5,
        )
        return response.choices[0].message.content.strip().lower().startswith("yes")


class Answering_Agent:
    def __init__(self, openai_client: OpenAI) -> None:
        self.client = openai_client

    def generate_response(
        self, query: str, docs: List[str], conv_history: List[Dict[str, str]], k: int = 5
    ) -> str:
        context = "\n---\n".join(docs[:k])
        system_msg = (
            "You are a helpful Machine Learning textbook assistant. "
            "Answer the user's question based primarily on the provided context from the textbook. "
            "If the question is follow-up, use conversation history for coherence. "
            "Only answer the ML-related part and ignore non-ML parts."
        )
        messages = [{"role": "system", "content": system_msg}]
        for m in conv_history[-6:]:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append(
            {
                "role": "user",
                "content": f"Context from textbook:\n{context}\n\nQuestion: {query}",
            }
        )
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()


class Relevant_Documents_Agent:
    def __init__(self, openai_client: OpenAI) -> None:
        self.client = openai_client

    def get_relevance(self, query: str, documents: List[str]) -> bool:
        docs_text = "\n---\n".join(documents[:5])
        system_msg = (
            "You are a relevance judge. Given a user query and retrieved document excerpts, "
            "determine if ANY documents contain useful info for the query.\n"
            "Respond with ONLY 'Yes' or 'No'."
        )
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Query: {query}\n\nDocuments:\n{docs_text}"},
            ],
            temperature=0,
            max_tokens=5,
        )
        return response.choices[0].message.content.strip().lower().startswith("yes")


class Head_Agent:
    def __init__(
        self,
        openai_key: str,
        pinecone_key: str,
        pinecone_index_name: str = INDEX_NAME,
        namespace: str = NAMESPACE,
    ) -> None:
        self.client = OpenAI(api_key=openai_key)
        pc = Pinecone(api_key=pinecone_key)
        self.pinecone_index = pc.Index(pinecone_index_name)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key,
        )
        self.namespace = namespace
        self.conversation_history: List[Dict[str, str]] = []
        self.setup_sub_agents()

    def setup_sub_agents(self):
        self.obnoxious_agent = Obnoxious_Agent(self.client)
        self.context_rewriter = Context_Rewriter_Agent(self.client)
        self.query_agent = Query_Agent(
            self.pinecone_index,
            self.client,
            self.embeddings,
            namespace=self.namespace,
        )
        self.relevance_agent = Relevant_Documents_Agent(self.client)
        self.answering_agent = Answering_Agent(self.client)

    def _is_greeting(self, query: str) -> bool:
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Determine if the following message is greeting/small talk. "
                        "Respond with ONLY 'Yes' or 'No'."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=5,
        )
        return response.choices[0].message.content.strip().lower().startswith("yes")

    def _handle_greeting(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly Machine Learning textbook assistant. "
                        "Reply warmly and briefly mention you can help with ML questions."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=128,
        )
        return response.choices[0].message.content.strip()

    def _extract_ml_part(self, query: str) -> Optional[str]:
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Detect whether the message is hybrid: contains both ML question and unrelated/offensive part.\n"
                        "If hybrid, return ONLY the ML part.\n"
                        "If not hybrid, return exactly 'NOT_HYBRID'."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=256,
        )
        result = response.choices[0].message.content.strip()
        if "NOT_HYBRID" in result:
            return None
        return result

    def _looks_like_followup(self, query: str) -> bool:
        lowered = query.lower().strip()
        followup_markers = [
            "it",
            "this",
            "that",
            "those",
            "these",
            "they",
            "them",
            "each",
            "which one",
            "tell me more",
            "more about",
            "in this context",
        ]
        if any(marker in lowered for marker in followup_markers):
            return True
        return len(lowered.split()) <= 8

    def _last_user_query(self) -> str:
        for item in reversed(self.conversation_history):
            if item.get("role") == "user":
                return item.get("content", "")
        return ""

    def process_query(self, query: str) -> Tuple[str, str]:
        if self.obnoxious_agent.check_query(query):
            ml_part = self._extract_ml_part(query)
            if ml_part is None:
                refusal = (
                    "I'm sorry, but I can't respond to messages containing offensive language. "
                    "Please rephrase respectfully."
                )
                self._save_turn(query, refusal)
                return refusal, "Obnoxious_Agent -> Refused"
            query_for_processing = ml_part
            is_hybrid = True
        else:
            query_for_processing = query
            is_hybrid = False

        if not is_hybrid and self._is_greeting(query_for_processing):
            greeting_resp = self._handle_greeting(query_for_processing)
            self._save_turn(query, greeting_resp)
            return greeting_resp, "Head_Agent -> Greeting"

        if not is_hybrid:
            ml_part = self._extract_ml_part(query_for_processing)
            if ml_part is not None:
                query_for_processing = ml_part
                is_hybrid = True

        rewritten = query_for_processing
        if self.conversation_history:
            rewritten = self.context_rewriter.rephrase(
                self.conversation_history, query_for_processing
            )

        followup_mode = bool(self.conversation_history) and self._looks_like_followup(
            query_for_processing
        )
        retrieval_query = rewritten
        if followup_mode:
            previous_user_query = self._last_user_query()
            if previous_user_query:
                retrieval_query = (
                    f"Previous user topic: {previous_user_query}\n"
                    f"Current follow-up question: {rewritten}"
                )

        if not self.query_agent.is_relevant_topic(retrieval_query):
            refusal = (
                "I'm sorry, that doesn't seem related to machine learning or textbook topics. "
                "I can only help with ML-related questions."
            )
            self._save_turn(query, refusal)
            return refusal, "Query_Agent -> Off-topic Refused"


        docs = self.query_agent.query_vector_store(retrieval_query, k=5)
        doc_texts = [doc.page_content for doc in docs]
        docs_relevant = self.relevance_agent.get_relevance(retrieval_query, doc_texts)
        if not docs_relevant and not followup_mode:
            refusal = (
                "I retrieved documents, but they don't seem relevant to your question. "
                "Please try rephrasing with more ML-specific detail."
            )
            self._save_turn(query, refusal)
            return refusal, "Relevant_Documents_Agent -> No relevant docs"

        answer = self.answering_agent.generate_response(
            rewritten, doc_texts, self.conversation_history
        )
        self._save_turn(query, answer)
        path = "Query_Agent -> Relevant_Documents_Agent -> Answering_Agent"
        if is_hybrid:
            path = "Hybrid_Extraction -> " + path
        return answer, path

    def chat(self, query: str):
        response, path_str = self.process_query(query)
        return response, path_str.split(" -> ")

    def _save_turn(self, user_query: str, response: str):
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": response})

    def reset_conversation(self):
        self.conversation_history = []


st.set_page_config(page_title="MP2 Part3 Chatbot", page_icon="🤖", layout="centered")
st.title("Mini Project 2 - Part 3 Chatbot")
st.caption("Multi-agent ML chatbot (OpenAI + Pinecone)")

def reset_conversation_state():
    if "head_agent" in st.session_state:
        st.session_state.head_agent.reset_conversation()
    st.session_state.messages = []
    st.rerun()

with st.sidebar:
    st.header("Settings")
    user_openai_key = st.text_input("OpenAI API Key", value=OPENAI_KEY, type="password")
    user_pinecone_key = st.text_input(
        "Pinecone API Key", value=PINECONE_KEY, type="password"
    )
    if st.button("Reset Conversation"):
        reset_conversation_state()

if "messages" not in st.session_state:
    st.session_state.messages = []

agent_key = f"{user_openai_key}_{user_pinecone_key}"
if (
    "head_agent" not in st.session_state
    or st.session_state.get("agent_key") != agent_key
):
    if user_openai_key and user_pinecone_key:
        with st.spinner("Initializing agents..."):
            st.session_state.head_agent = Head_Agent(
                user_openai_key,
                user_pinecone_key,
                pinecone_index_name=INDEX_NAME,
                namespace=NAMESPACE,
            )
            st.session_state.agent_key = agent_key
    else:
        st.warning("Please provide OpenAI and Pinecone keys in the sidebar.")
        st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "agent_path" in message:
            st.caption("Agent path: " + " -> ".join(message["agent_path"]))

if prompt := st.chat_input("Ask me an ML question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response, agent_path = st.session_state.head_agent.chat(prompt)
            except Exception as e:
                response = f"Error: {e}"
                agent_path = []

        st.markdown(response)
        if agent_path:
            st.caption("Agent path: " + " -> ".join(agent_path))

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "agent_path": agent_path}
    )