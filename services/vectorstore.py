from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from functools import lru_cache
from typing import List, Dict, Optional
import os

VECTOR_DB_PATH = "data/vectorstore"

os.makedirs(VECTOR_DB_PATH, exist_ok=True)

@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    print("🔄 Loading embedding model...")
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

_vectorstore = None

def get_vectorstore() -> Chroma:
    global _vectorstore

    if _vectorstore is None:
        print("🔄 Loading vectorstore...")
        embedding_model = get_embedding_model()

        _vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embedding_model
        )

    return _vectorstore


def file_already_indexed(file_id: str) -> bool:
    vs = get_vectorstore()
    results = vs.get(where={"file_id": file_id}, limit=1)
    return len(results["ids"]) > 0


def store_chunks(chunks: List[Dict]) -> None:
    if not chunks:
        return

    file_id = chunks[0]["file_id"]

    if file_already_indexed(file_id):
        print(f"'{file_id}' already indexed — skipping.")
        return

    vectorstore = get_vectorstore()

    texts = []
    metadatas = []
    ids = []

    for c in chunks:
        texts.append(c["content"])
        metadatas.append(c)
        ids.append(c["chunk_id"])

    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )

    vectorstore.persist()
    print(f"✅  Stored {len(chunks)} chunks for '{file_id}'")


def delete_file(file_id: str) -> None:
    vs = get_vectorstore()
    vs.delete(where={"file_id": file_id})
    vs.persist()
    print(f"Deleted all chunks for '{file_id}'")


def search_chunks(query: str, top_k: int = 3, file_id: Optional[str] = None):
    vectorstore = get_vectorstore()

    kwargs = {"k": top_k}
    if file_id:
        kwargs["filter"] = {"file_id": file_id}

    return vectorstore.similarity_search(query, **kwargs)