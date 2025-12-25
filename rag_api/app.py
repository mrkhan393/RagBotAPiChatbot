from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from rag_api.schemas import QueryRequest
from rag_api.query import query_rag
from rag_api.ingest import ingest_uploaded_file
from typing import List

app = FastAPI(title="RAG Document Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload Endpoint
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded_info = []
    for file in files:
        file_id = await ingest_uploaded_file(file)
        uploaded_info.append({"file_id": file_id, "filename": file.filename})
    return {"uploaded_files": uploaded_info}

# -------------------- Query Endpoint --------------------
@app.post("/query")
async def query(req: QueryRequest):
    try:
        answer, context, sources, ocr_text = query_rag(req.question, req.image_base64)
        return {
            "answer": answer,
            "context": context,
            "sources": sources,
            "ocr_text": ocr_text
        }
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "context": "", "sources": [], "ocr_text": ""}
