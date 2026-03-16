"""
Multi-Agent Chatbot System for Machine Learning Textbook Q&A.
Uses OpenAI GPT-4.1-nano and Pinecone vector store.
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


# ---------------------------------------------------------------------------
# Obnoxious Agent  (NO Langchain)
# ---------------------------------------------------------------------------
class Obnoxious_Agent:
    """Detects whether a user query contains obnoxious / offensive content."""

    def __init__(self, client: OpenAI) -> None:
        self.client = client
        self.prompt = (
            "You are a content-moderation classifier. "
            "Determine whether the following user message is obnoxious, rude, "
            "offensive, or contains hate speech / personal attacks.\n"
            "Respond with ONLY 'Yes' if the message is obnoxious, or 'No' if it is not.\n"
            "Do not provide any explanation."
        )

    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def extract_action(self, response_text: str) -> bool:
        """Return True if the query IS obnoxious."""
        return response_text.strip().lower().startswith("yes")

    def check_query(self, query: str) -> bool:
        """Return True if obnoxious, False otherwise."""
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=5,
        )
        answer = response.choices[0].message.content
        return self.extract_action(answer)


# ---------------------------------------------------------------------------
# Context Rewriter Agent
# ---------------------------------------------------------------------------
class Context_Rewriter_Agent:
    """Rewrites the latest user query by resolving coreferences from history."""

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
            "- Replace ALL pronouns (it, that, this, they, them) with the specific ML "
            "concepts they refer to from the conversation history.\n"
            "- If the user says 'tell me more', 'explain further', 'give an example', "
            "make the rewritten query explicitly mention the topic being discussed.\n"
            "- If the message is already self-contained, return it unchanged.\n"
            "- Return ONLY the rewritten query, nothing else."
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


# ---------------------------------------------------------------------------
# Query Agent  (Langchain allowed)
# ---------------------------------------------------------------------------
class Query_Agent:
    """Queries Pinecone vector store and decides whether topic is relevant."""

    def __init__(
        self,
        pinecone_index,
        openai_client: OpenAI,
        embeddings: OpenAIEmbeddings,
        namespace: str = "ns2500",
    ) -> None:
        self.index = pinecone_index
        self.client = openai_client
        self.embeddings = embeddings
        self.namespace = namespace
        self.vectorstore = PineconeVectorStore(
            index=pinecone_index,
            embedding=embeddings,
            text_key="text",
            namespace=namespace,
        )
        self.prompt = (
            "You are a relevance classifier for a Machine Learning textbook assistant. "
            "Given the user query, determine if it is related to Machine Learning, "
            "AI, data science, statistics, or the topics typically covered in an ML textbook. "
            "Respond with ONLY 'Yes' if relevant, or 'No' if not relevant."
        )

    def query_vector_store(self, query: str, k: int = 5) -> List:
        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def extract_action(self, response_text: str, query: str = None) -> bool:
        """Return True if the query is relevant to the ML textbook."""
        return response_text.strip().lower().startswith("yes")

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
        return self.extract_action(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Relevant Documents Agent  (NO Langchain)
# ---------------------------------------------------------------------------
class Relevant_Documents_Agent:
    """Judges whether the retrieved documents are relevant to the user query."""

    def __init__(self, openai_client: OpenAI) -> None:
        self.client = openai_client

    def get_relevance(self, query: str, documents: List[str]) -> bool:
        """Return True if at least some docs are relevant to the query."""
        docs_text = "\n---\n".join(documents[:5])
        system_msg = (
            "You are a relevance judge. Given a user query and a set of retrieved "
            "document excerpts, determine if ANY of the documents contain information "
            "useful for answering the query.\n"
            "Respond with ONLY 'Yes' if relevant documents exist, or 'No' if none are relevant."
        )
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nDocuments:\n{docs_text}",
                },
            ],
            temperature=0,
            max_tokens=5,
        )
        return response.choices[0].message.content.strip().lower().startswith("yes")


# ---------------------------------------------------------------------------
# Answering Agent
# ---------------------------------------------------------------------------
class Answering_Agent:
    """Generates final answers using retrieved context."""

    def __init__(self, openai_client: OpenAI) -> None:
        self.client = openai_client

    def generate_response(
        self,
        query: str,
        docs: List[str],
        conv_history: List[Dict[str, str]],
        k: int = 5,
    ) -> str:
        context = "\n---\n".join(docs[:k])
        system_msg = (
            "You are a helpful Machine Learning textbook assistant. "
            "Answer the user's question based primarily on the provided context from the textbook. "
            "Use the context as your main source, but if the context is partially relevant, "
            "still provide the best possible answer using what is available. "
            "Only say you cannot answer if the context is completely unrelated to the question. "
            "Be concise, accurate, and helpful. "
            "If the question is a follow-up from a previous conversation, use the "
            "conversation history to provide a coherent and contextual answer. "
            "ONLY answer the Machine Learning related part of the question. "
            "Ignore any non-ML parts (e.g., jokes, unrelated questions, personal opinions)."
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


# ---------------------------------------------------------------------------
# Head Agent  (Controller)
# ---------------------------------------------------------------------------
class Head_Agent:
    """
    Orchestrates all sub-agents to handle user queries.

    Pipeline:
      1. Check if query is obnoxious  -> refuse
      2. Rewrite query for context     -> disambiguate
      3. Check topic relevance         -> refuse if off-topic
      4. Retrieve documents            -> Pinecone
      5. Check document relevance      -> refuse if no useful docs
      6. Generate answer               -> final response
    """

    def __init__(
        self,
        openai_key: str,
        pinecone_key: str,
        pinecone_index_name: str = "machine-learning-textbook",
        namespace: str = "ns2500",
    ) -> None:
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.index_name = pinecone_index_name
        self.namespace = namespace

        self.client = OpenAI(api_key=openai_key)
        pc = Pinecone(api_key=pinecone_key)
        self.pinecone_index = pc.Index(pinecone_index_name)

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key,
        )

        self.conversation_history: List[Dict[str, str]] = []
        self.setup_sub_agents()

    def setup_sub_agents(self):
        self.obnoxious_agent = Obnoxious_Agent(self.client)
        self.context_rewriter = Context_Rewriter_Agent(self.client)
        self.query_agent = Query_Agent(
            self.pinecone_index, self.client, self.embeddings, self.namespace
        )
        self.relevance_agent = Relevant_Documents_Agent(self.client)
        self.answering_agent = Answering_Agent(self.client)

    def _is_greeting(self, query: str) -> bool:
        """Quick check for greetings / small talk."""
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Determine if the following message is a greeting, small talk, "
                        "or a general conversational opener that does NOT ask a specific "
                        "knowledge question. This includes:\n"
                        "- Greetings: 'hello', 'hi', 'hey', 'good morning'\n"
                        "- Small talk: 'how are you?', 'what's up?'\n"
                        "- General requests: 'can you help me?', 'I need help'\n"
                        "- Fun/casual: 'do you have any fun facts?', 'tell me something interesting'\n"
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
                        "Respond to the greeting or casual message warmly and briefly "
                        "mention that you can help with machine learning questions."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=128,
        )
        return response.choices[0].message.content.strip()

    def _extract_ml_part(self, query: str) -> Optional[str]:
        """
        Detect if a query is hybrid (contains both ML and non-ML parts).
        If so, extract ONLY the ML-relevant portion. Returns None if not hybrid.
        """
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intent classifier for a Machine Learning textbook assistant.\n"
                        "Analyze the user message and determine if it contains BOTH:\n"
                        "  (a) A Machine Learning / AI / data science question, AND\n"
                        "  (b) An unrelated or obnoxious component.\n\n"
                        "If YES (it is a hybrid message), extract ONLY the ML-relevant question "
                        "and return it. Strip away everything unrelated.\n"
                        "If NO (the message is purely ML or purely non-ML), respond with "
                        "exactly 'NOT_HYBRID'.\n\n"
                        "Return ONLY the extracted ML question or 'NOT_HYBRID'. Nothing else."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=256,
        )
        result = response.choices[0].message.content.strip()
        if result == "NOT_HYBRID" or "NOT_HYBRID" in result:
            return None
        return result

    def process_query(self, query: str) -> Tuple[str, str]:
        """
        Process a single user query through the agent pipeline.

        Returns:
            (response_text, agent_path) where agent_path describes which agents were used.
        """
        # Step 1: Obnoxious check (only for purely obnoxious messages)
        if self.obnoxious_agent.check_query(query):
            # Before refusing, check if there's also an ML part (hybrid)
            ml_part = self._extract_ml_part(query)
            if ml_part is None:
                self.conversation_history.append({"role": "user", "content": query})
                refusal = "I'm sorry, but I can't respond to messages that contain offensive or inappropriate language. Please rephrase your question respectfully."
                self.conversation_history.append({"role": "assistant", "content": refusal})
                return refusal, "Obnoxious_Agent -> Refused"
            # Has ML part — continue with extracted ML question
            query_for_processing = ml_part
            is_hybrid = True
        else:
            is_hybrid = False
            query_for_processing = query

        # Step 2: Check for greeting
        if not is_hybrid and self._is_greeting(query_for_processing):
            greeting_resp = self._handle_greeting(query_for_processing)
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": greeting_resp})
            return greeting_resp, "Head_Agent -> Greeting"

        # Step 2.5: Check for hybrid (non-obnoxious hybrid)
        if not is_hybrid:
            ml_part = self._extract_ml_part(query_for_processing)
            if ml_part is not None:
                query_for_processing = ml_part
                is_hybrid = True

        # Step 3: Rewrite query using conversation context
        if len(self.conversation_history) > 0:
            rewritten = self.context_rewriter.rephrase(
                self.conversation_history, query_for_processing
            )
        else:
            rewritten = query_for_processing

        # Step 4: Check topic relevance
        if not self.query_agent.is_relevant_topic(rewritten):
            self.conversation_history.append({"role": "user", "content": query})
            refusal = "I'm sorry, but that question doesn't seem to be related to Machine Learning or the topics covered in the textbook. I can only help with ML-related questions."
            self.conversation_history.append({"role": "assistant", "content": refusal})
            return refusal, "Query_Agent -> Off-topic Refused"

        # Step 5: Retrieve documents
        docs = self.query_agent.query_vector_store(rewritten, k=5)
        doc_texts = [doc.page_content for doc in docs]

        # Step 6: Check document relevance
        if not self.relevance_agent.get_relevance(rewritten, doc_texts):
            self.conversation_history.append({"role": "user", "content": query})
            refusal = "I found some documents but none of them seem relevant to your question. Could you please rephrase or ask a different ML-related question?"
            self.conversation_history.append({"role": "assistant", "content": refusal})
            return refusal, "Relevant_Documents_Agent -> No relevant docs"

        # Step 7: Generate answer
        answer = self.answering_agent.generate_response(
            rewritten, doc_texts, self.conversation_history
        )
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": answer})
        path = "Query_Agent -> Relevant_Documents_Agent -> Answering_Agent"
        if is_hybrid:
            path = "Hybrid_Extraction -> " + path
        return answer, path

    def reset_conversation(self):
        self.conversation_history = []

    def main_loop(self):
        """Interactive terminal loop (for testing)."""
        print("=== ML Textbook Chatbot ===")
        print("Type 'quit' to exit, 'reset' to clear history.\n")
        while True:
            query = input("You: ").strip()
            if query.lower() == "quit":
                break
            if query.lower() == "reset":
                self.reset_conversation()
                print("Conversation reset.\n")
                continue
            response, path = self.process_query(query)
            print(f"\nBot: {response}")
            print(f"  [Agent path: {path}]\n")
