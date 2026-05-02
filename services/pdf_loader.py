import pdfplumber
import hashlib
import os
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime


def _file_id(file_path: str) -> str:
    stem = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()[:6]
    return f"{stem}__{digest}"


def extract_text_tables(file_path: str) -> List[Dict]:
    documents = []
    file_id = _file_id(file_path)
    file_name = os.path.basename(file_path)

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):

            # -------- TEXT --------
            text = page.extract_text()
            if text:
                documents.append({
                    "content": text,
                    "page": page_num + 1,
                    "type": "text"
                })

            # -------- TABLES --------
            tables = page.extract_tables()
            for table in tables:
                table_text = "\n".join(
                    [" | ".join([str(cell or "") for cell in row]) for row in table]
                )
                if table_text.strip():
                    documents.append({
                        "content": table_text,
                        "page": page_num + 1,
                        "type": "table"
                    })

    # -------- CHUNKING --------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=130
    )

    structured_chunks = []

    for doc in documents:
        chunks = splitter.split_text(doc["content"])

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{file_id}__p{doc['page']}_{doc['type']}_c{idx}"

            structured_chunks.append({
                "chunk_id":   chunk_id,
                "file_id":    file_id,
                "content":    chunk,
                "source":     file_path,
                "file_name":  file_name,
                "page":       doc["page"],
                "type":       doc["type"],
                "created_at": datetime.now().isoformat()
            })

    print(file_id, file_name, len(structured_chunks));

    return structured_chunks