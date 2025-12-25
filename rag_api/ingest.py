import os
import uuid
from fastapi import UploadFile
from rag_api.vectorstore import add_to_vectorstore
import pandas as pd
import fitz  
from docx import Document
from utils.parsers import parse_file

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def ingest_uploaded_file(file: UploadFile):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, file_id + "_" + file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text
    text = parse_file(file_path)

    # chunk text
    from utils.chunking import chunk_text
    text_chunks = chunk_text(text)

    # Add to vectorstore with metadata
    add_to_vectorstore(text_chunks, metadata={"filename": file.filename})

    return file_id
