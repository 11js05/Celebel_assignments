"""in this week, we will implement a simple RAG (retrieval-augmented generation) pipeline 
using Google Gemini 2.5 for document question answering. The pipeline will allow users to 
upload documents (PDF or TXT), process them into chunks, embed the chunks, and then answer 
questions based on the content of those documents."""
import os
import io
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import faiss
from pypdf import PdfReader
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

class DocumentProcessor:

    @staticmethod
    def load_pdf(file_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    @staticmethod
    def load_txt(file_bytes: bytes) -> str:
        return file_bytes.decode("utf-8", errors="ignore")

    @classmethod
    def load(cls, filename: str, file_bytes: bytes) -> str:
        ext = filename.lower().split(".")[-1]
        if ext == "pdf":
            return cls.load_pdf(file_bytes)
        elif ext in ("txt", "md"):
            return cls.load_txt(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

class TextChunker:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        text = text.replace("\r\n", "\n").strip()
        if not text:
            return []
        chunks = self._recursive_split(text, ["\n\n", "\n", ". ", " "])
        return self._merge_with_overlap(chunks)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = separators[0]
        parts = text.split(sep)
        results, buffer = [], ""
        for part in parts:
            candidate = (buffer + sep + part) if buffer else part
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    results.append(buffer)
                if len(part) > self.chunk_size:
                    results.extend(self._recursive_split(part, separators[1:]))
                    buffer = ""
                else:
                    buffer = part
        if buffer:
            results.append(buffer)
        return results

    def _merge_with_overlap(self, chunks: List[str]) -> List[str]:
        if not chunks:
            return []
        final = [chunks[0].strip()]
        for chunk in chunks[1:]:
            prev = final[-1]
            overlap_text = prev[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
            final.append((overlap_text + " " + chunk).strip())
        return [c for c in final if c]

class GeminiEmbeddings:

    def __init__(self, api_key: str, model: str = "models/gemini-embedding-001"):
        genai.configure(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            resp = genai.embed_content(model=self.model, content=t, task_type="retrieval_document")
            vectors.append(resp["embedding"])
        return np.array(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        resp = genai.embed_content(model=self.model, content=text, task_type="retrieval_query")
        return np.array(resp["embedding"], dtype="float32")

class VectorStore:

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vecs / norms

    def add(self, embeddings: np.ndarray, texts: List[str], metadatas: List[Dict[str, Any]]):
        embeddings = self._normalize(embeddings)
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def search(self, query_embedding: np.ndarray, k: int = 4) -> List[Tuple[str, Dict[str, Any], float]]:
        if self.index.ntotal == 0:
            return []
        q = self._normalize(query_embedding.reshape(1, -1))
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(q, k)
        return [
            (self.texts[idx], self.metadatas[idx], float(score))
            for score, idx in zip(scores[0], idxs[0]) if idx != -1
        ]

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "store.pkl"), "wb") as f:
            pickle.dump({"texts": self.texts, "metadatas": self.metadatas, "dim": self.dim}, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        with open(os.path.join(path, "store.pkl"), "rb") as f:
            data = pickle.load(f)
        store = cls(data["dim"])
        store.index = faiss.read_index(os.path.join(path, "index.faiss"))
        store.texts = data["texts"]
        store.metadatas = data["metadatas"]
        return store

@dataclass
class RAGConfig:
    api_key: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "models/gemini-embedding-001"
    gen_model: str = "gemini-2.5-flash"
    top_k: int = 4

class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        genai.configure(api_key=config.api_key)

        self.chunker = TextChunker(config.chunk_size, config.chunk_overlap)
        self.embedder = GeminiEmbeddings(config.api_key, config.embedding_model)
        self.generator = genai.GenerativeModel(config.gen_model)

        self.vector_store: Optional[VectorStore] = None
        self.doc_names: List[str] = []

    def ingest(self, files: List[Tuple[str, bytes]]) -> int:
        all_chunks, all_metas = [], []

        for filename, file_bytes in files:
            text = DocumentProcessor.load(filename, file_bytes)
            chunks = self.chunker.split(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metas.append({"source": filename, "chunk_id": i})
            self.doc_names.append(filename)

        if not all_chunks:
            raise ValueError("No text extracted from the uploaded document(s).")

        embeddings = self.embedder.embed_documents(all_chunks)
        if self.vector_store is None:
            self.vector_store = VectorStore(embeddings.shape[1])
        self.vector_store.add(embeddings, all_chunks, all_metas)

        return len(all_chunks)

    def answer(self, question: str, k: Optional[int] = None) -> Dict[str, Any]:
        if self.vector_store is None or self.vector_store.index.ntotal == 0:
            raise ValueError("No documents ingested yet. Upload and process documents first.")

        k = k or self.config.top_k
        query_vec = self.embedder.embed_query(question)
        results = self.vector_store.search(query_vec, k=k)

        context = "\n\n---\n\n".join(
            f"[Source: {meta['source']} | chunk {meta['chunk_id']}]\n{text}"
            for text, meta, _ in results
        )

        prompt = (
            "You are a helpful assistant answering questions using ONLY the context below.\n"
            "If the answer is not contained in the context, say you don't have enough information, "
            "do not make anything up.\n\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "ANSWER (be concise and grounded strictly in the context above):"
        )
        response = self.generator.generate_content(prompt)

        return {
            "answer": response.text,
            "sources": [
                {"source": meta["source"], "chunk_id": meta["chunk_id"], "score": score, "text": text}
                for text, meta, score in results
            ],
        }

st.set_page_config(page_title="RAG Document QA (Gemini 2.5)", layout="wide")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input(
        "Google API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Get one at https://aistudio.google.com/app/apikey",
    )

    gen_model = st.selectbox("Generation model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)

    with st.expander("Advanced settings"):
        chunk_size = st.slider("Chunk size (chars)", 300, 3000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap (chars)", 0, 500, 200, step=50)
        top_k = st.slider("Chunks retrieved per query (k)", 1, 10, 4)

    st.divider()
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt", "md"], accept_multiple_files=True
    )

    if st.button("Process documents", use_container_width=True):
        if not api_key:
            st.error("Enter your Google API key first.")
        elif not uploaded_files:
            st.error("Upload at least one document.")
        else:
            with st.spinner("Chunking, embedding, and indexing documents..."):
                try:
                    config = RAGConfig(
                        api_key=api_key,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        gen_model=gen_model,
                        top_k=top_k,
                    )
                    st.session_state.pipeline = RAGPipeline(config)
                    files_payload = [(f.name, f.read()) for f in uploaded_files]
                    num_chunks = st.session_state.pipeline.ingest(files_payload)
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    st.success(f"Indexed {num_chunks} chunks from {len(uploaded_files)} file(s).")
                except Exception as e:
                    st.error(f"Error while processing: {e}")

    if st.session_state.processed_files:
        st.divider()
        st.caption("Indexed files:")
        for name in st.session_state.processed_files:
            st.write(f"• {name}")

st.title("Document Question Answering (RAG)")
st.caption("Ask questions grounded in your uploaded documents — powered by Google Gemini 2.5.")

if not st.session_state.pipeline:
    st.info("Upload documents and click **Process documents** to get started.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['source']}** (chunk {s['chunk_id']}, score {s['score']:.3f})")
                        st.text(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))

    question = st.chat_input("Ask a question about your documents...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    result = st.session_state.pipeline.answer(question)
                    st.markdown(result["answer"])
                    with st.expander("Sources"):
                        for s in result["sources"]:
                            st.markdown(f"**{s['source']}** (chunk {s['chunk_id']}, score {s['score']:.3f})")
                            st.text(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                    )
                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    st.session_state.chat_history.append({"role": "assistant", "content": err})
