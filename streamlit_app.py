import streamlit as st
import requests
import base64

st.set_page_config(page_title="RAG Document Intelligence", layout="wide")

# -------------------- Sidebar --------------------
st.sidebar.title("⚙️ Mode Selection")
mode = st.sidebar.radio(
    "Select Mode",
    ["Document Q&A", "OCR Image Q&A"]
)

# -------------------- Title --------------------
st.markdown("""
# RAG Document Intelligence System
Ask questions across PDFs, Word files, images, datasets, and databases using Retrieval-Augmented Generation.
""")

API_URL = "http://127.0.0.1:8000"


# -------------------- Document Mode (Q&A) --------------------

if mode == "Document Q&A":

    st.subheader("📁 Upload Document(s)")
    uploaded_files = st.file_uploader(
        "Drag and drop files here (PDF, DOCX, TXT, JPG, JPEG, PNG, CSV, DB)",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "csv", "db"],
        accept_multiple_files=True
    )

    if uploaded_files:
        col1, col2 = st.columns(2)

        for f in uploaded_files:
            file_bytes = f.read()

            files_payload = [("files", (f.name, file_bytes, f.type))]
            try:
                res = requests.post(f"{API_URL}/upload", files=files_payload)
                res.raise_for_status()
                uploaded_data = res.json()
                file_id = uploaded_data["uploaded_files"][0]["file_id"]
                col2.success(f"Uploaded {f.name} with ID {file_id}")
            except Exception as e:
                col2.error(f"Upload failed: {f.name} — {e}")

    # -------------------- Question Section --------------------
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question here:")

    if st.button("Get Answer") and question.strip():
        try:
            res = requests.post(
                f"{API_URL}/query",
                json={"question": question.strip()}
            )
            res.raise_for_status()
            data = res.json()

            # ---------- Deduplicate Context ----------
            seen = set()
            clean_context = []
            for line in data.get("context", "").split("\n"):
                if line.strip() and line not in seen:
                    clean_context.append(line)
                    seen.add(line)

            st.markdown("### Answer")
            st.info(data.get("answer", ""))

            st.markdown("### 📄 Retrieved Context")
            if clean_context:
                st.text_area(
                    "Context used for LLM",
                    "\n".join(clean_context),
                    height=250
                )
            else:
                st.write("No context retrieved.")

            # ---------- Deduplicate Sources ----------
            st.markdown("### 📌 Sources")
            seen_sources = set()
            for s in data.get("sources", []):
                key = (s.get("filename"), s.get("chunk_index"))
                if key not in seen_sources:
                    st.write(s)
                    seen_sources.add(key)

            if not seen_sources:
                st.write("No sources available.")

        except Exception as e:
            st.error(f"Query failed: {e}")

# -------------------- OCR IMAGE Q&A MODE --------------------

else:
    st.subheader("🖼️ Upload Image for OCR")

    ocr_file = st.file_uploader(
        "Drag and drop a single image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

    if ocr_file:
        image_bytes = ocr_file.read()
        st.image(image_bytes, caption=ocr_file.name, width=300)

        image_base64 = base64.b64encode(image_bytes).decode()

        if st.button("🔍 Run OCR"):
            try:
                res = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": "Extract all readable text from this image",
                        "image_base64": image_base64
                    }
                )
                res.raise_for_status()
                data = res.json()

                st.markdown("### 🖼️ OCR Result")
                st.text_area(
                    "Extracted Text",
                    data.get("ocr_text", ""),
                    height=300
                )

            except Exception as e:
                st.error(f"OCR processing failed: {e}")
