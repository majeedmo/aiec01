from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import tiktoken
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

RAG_SYSTEM_PROMPT = """You are an educational feline-health information assistant.

Answer only from the retrieved context. If the context does not provide
enough information, say so. Do not diagnose, prescribe, or provide individualized
veterinary advice.
"""

RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 75
DEFAULT_RETRIEVAL_K = 3


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    chat_model: str
    embedding_model: str
    api_key: str
    api_base: str
    embedding_dimensions: int | None = None


class RAGState(TypedDict):
    question: str
    context: list[Document]
    answer: str


def _tiktoken_len(text: str) -> int:
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)


def fireworks_config() -> ProviderConfig:
    embedding_model = os.environ.get("FIREWORKS_EMBEDDING_MODEL", "")
    if not embedding_model.startswith("accounts/fireworks/"):
        embedding_model = "accounts/fireworks/models/qwen3-embedding-8b"

    return ProviderConfig(
        name="fireworks",
        chat_model=os.environ.get(
            "FIREWORKS_CHAT_MODEL", "accounts/fireworks/models/gpt-oss-20b"
        ),
        embedding_model=embedding_model,
        api_key=os.environ["FIREWORKS_API_KEY"],
        api_base=FIREWORKS_BASE_URL,
        embedding_dimensions=4096,
    )


def openai_config() -> ProviderConfig:
    return ProviderConfig(
        name="openai",
        chat_model=os.environ.get("OPENAI_RAG_MODEL", "gpt-4.1-mini"),
        embedding_model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.environ["OPENAI_API_KEY"],
        api_base=os.environ.get("OPENAI_BASE_URL", OPENAI_BASE_URL),
        embedding_dimensions=None,
    )


def build_embeddings(config: ProviderConfig) -> OpenAIEmbeddings:
    kwargs: dict[str, Any] = {
        "model": config.embedding_model,
        "api_key": config.api_key,
        "base_url": config.api_base,
        "check_embedding_ctx_length": False,
    }
    if config.embedding_dimensions is not None:
        kwargs["dimensions"] = config.embedding_dimensions
    return OpenAIEmbeddings(**kwargs)


def build_chat_model(config: ProviderConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.chat_model,
        api_key=config.api_key,
        base_url=config.api_base,
        temperature=0,
        max_retries=2,
        timeout=120,
    )


def load_corpus_documents(data_dir: str) -> list[Document]:
    directory_loader = DirectoryLoader(
        data_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader
    )
    return directory_loader.load()


def build_vector_store(
    documents: list[Document], config: ProviderConfig
) -> QdrantVectorStore:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        length_function=_tiktoken_len,
    )
    chunks = splitter.split_documents(documents)
    return QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=build_embeddings(config),
        location=":memory:",
        collection_name=f"rag_eval_{config.name}_{uuid4().hex[:8]}",
    )


def build_rag_graph(
    retriever,
    chat_model: ChatOpenAI,
) -> Any:
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "Question:\n{question}\n\nRetrieved context:\n{context}"),
        ]
    )

    def retrieve(state: RAGState) -> RAGState:
        return {"context": retriever.invoke(state["question"])}

    def generate(state: RAGState) -> RAGState:
        context_text = "\n\n".join(doc.page_content for doc in state["context"])
        messages = rag_prompt.format_messages(
            question=state["question"],
            context=context_text,
        )
        response = chat_model.invoke(messages)
        return {"answer": str(response.content)}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    return graph.compile()


def build_provider_rag_pipeline(
    data_dir: str,
    config: ProviderConfig,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
) -> Any:
    documents = load_corpus_documents(data_dir)
    vector_store = build_vector_store(documents, config)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retrieval_k},
    )
    chat_model = build_chat_model(config)
    return build_rag_graph(retriever, chat_model)
