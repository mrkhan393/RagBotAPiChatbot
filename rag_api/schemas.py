from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    image_base64: str = None  # optional
