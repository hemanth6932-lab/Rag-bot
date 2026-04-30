import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from core.llm import generate_answer
from services.pdf_loader import extract_text_tables
from services.vectorstore import delete_file, file_already_indexed, store_chunks, search_chunks

router = APIRouter()

UPLOAD_DIR = "data/files"


#Upload PDF View
@router.post("/upload")
async def upload_pdf_view(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    chunks = extract_text_tables(file_path)
    store_chunks(chunks)

    return {
        "message": "PDF uploaded and indexed successfully",
        "chunks": len(chunks)
    }

@router.get("/search")
def search(q: str):
    results = search_chunks(q)

    return [
        {
            "content": r.page_content,
            "metadata": r.metadata
        }
        for r in results
    ]

@router.get("/Ask")
def ask(q: str):
    results = search_chunks(q, top_k=3)

    context = ""
    for i, r in enumerate(results):
        context += f"Chunk {i+1}:\n{r.page_content}\n\n"

    answer = generate_answer(q, context)

    return {
        "question": q,
        "answer": answer
    }

# Delete PDF View
@router.delete("/delete/{file_id}")
def delete_pdf_view(file_id: str):
    if not file_already_indexed(file_id):
        raise HTTPException(status_code=404, detail=f"'{file_id}' not found in vectorstore")

    delete_file(file_id)

    return {
        "message": f"'{file_id}' deleted successfully"}