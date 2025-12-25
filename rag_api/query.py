from rag_api.llm import generate_answer
from rag_api.vectorstore import search_vectorstore
from rag_api.ocr_utils import ocr_image

def query_rag(question: str = None, image_base64: str = None, mode: str = "document"):
    """
    mode: "document" -> standard RAG query
          "ocr" -> OCR-only mode
    """

    context = ""
    sources = []
    ocr_text = ""

    # OCR mode
    if mode == "ocr" and image_base64:
        ocr_text = ocr_image(image_base64)
        context = ocr_text  
        sources = [{"filename": "ocr_image"}]
        answer = ocr_text  
        return answer, context, sources, ocr_text

    # If image is provided, perform OCR and add to context
    if image_base64:
        ocr_text = ocr_image(image_base64)
        context += ocr_text + "\n"
        sources.append({"filename": "ocr_image"})

    # Vector search
    if question:
        search_results = search_vectorstore(question)
        for r in search_results:
            context += r["text"] + "\n"
            sources.append(r["metadata"])

        # Generate answer from LLM
        if context.strip():
            answer = generate_answer(context, question)
        else:
            answer = "No relevant context found for your question."
    else:
        answer = "No question provided."

    return answer, context, sources, ocr_text
