"""
ingest.py -- Reads all .txt files from backend/data/, chunks them,
embeds them using SentenceTransformers, and stores them in a
local ChromaDB vector database.

Run once:  python ingest.py
"""

import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# -- Config -----------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def ingest():
    """Load, chunk, embed and persist documents."""

    # 1. Gather all .txt files
    txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    if not txt_files:
        print("[WARN] No .txt files found in", DATA_DIR)
        return

    print(f"[FILES] Found {len(txt_files)} file(s): {[os.path.basename(f) for f in txt_files]}")

    # 2. Load documents
    all_docs = []
    for fpath in txt_files:
        try:
            loader = TextLoader(fpath, encoding="utf-8")
            docs = loader.load()
            # Tag each doc with its source filename
            for doc in docs:
                doc.metadata["source"] = os.path.basename(fpath)
            all_docs.extend(docs)
            print(f"  + Loaded {os.path.basename(fpath)} ({len(docs)} doc(s))")
        except Exception as e:
            print(f"  x Error loading {os.path.basename(fpath)}: {e}")

    # 3. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"[SPLIT] Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # 4. Embed & persist to ChromaDB
    print(f"[EMBED] Loading embedding model: {EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Delete old database if it exists
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("[CLEAN] Cleared old ChromaDB")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"[DONE] Ingestion complete! {len(chunks)} chunks stored in {CHROMA_DIR}")
    return vectorstore


if __name__ == "__main__":
    ingest()
