import pdfplumber, docx, pandas as pd, sqlite3
from PIL import Image
import pytesseract

def parse_file(path):
    if path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    if path.endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])

    if path.endswith(".txt"):
        return open(path, encoding="utf-8").read()

    if path.endswith((".png", ".jpg", ".jpeg")):
        return pytesseract.image_to_string(Image.open(path))

    if path.endswith(".csv"):
        return pd.read_csv(path).to_string()

    if path.endswith(".db"):
        conn = sqlite3.connect(path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        text = ""
        for t in tables["name"]:
            text += pd.read_sql(f"SELECT * FROM {t}", conn).to_string()
        return text

    raise ValueError("Unsupported file type")
