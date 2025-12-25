import os
from openai import OpenAI
from openai._exceptions import RateLimitError, OpenAIError
import dotenv

dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(context: str, question: str, ocr_text: str = None):
    """
    Generates a structured answer from LLM based on context, optional OCR, and user question.
    """
    # Combine OCR text with retrieved context if available
    full_context = ""
    if ocr_text:
        full_context += f"Text from OCR image:\n{ocr_text}\n\n"
    if context:
        full_context += f"Relevant document context:\n{context}\n"

    # Construct a strong system + user prompt
    system_prompt = (
        "You are an expert assistant specialized in analyzing biomedical, chemical, "
        "and scientific documents. Your job is to answer questions accurately using the "
        "context provided, without adding any information not present in the documents."
    )

    user_prompt = (
        f"{full_context}\n"
        f"User Question: {question}\n\n"
        "Instructions:\n"
        "- Answer clearly and concisely.\n"
        "- Cite sources (filename, page, chunk index) for each fact.\n"
        "- Use bullet points if multiple points exist.\n"
        "- If the answer is not in the context, reply 'Information not available in the provided documents.'"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()

    except RateLimitError:
        return "OpenAI API quota exceeded. Please try again later."
    except OpenAIError as e:
        return f"OpenAI API error: {str(e)}"
    except Exception as e:
        return f"LLM error: {str(e)}"
