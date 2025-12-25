# rag_api/ocr_utils.py
import base64
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import cv2
import pytesseract

def preprocess_image_for_ocr(image_bytes: bytes) -> Image.Image:
    """
    Preprocess the image to improve OCR accuracy:
    - Convert to grayscale
    - Apply thresholding (binarization)
    - Optional: Dilate/Erode to remove noise
    """
    # Load image
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Convert to grayscale
    gray = np.array(img.convert("L"))
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional: remove small noise
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)
    processed = cv2.erode(processed, kernel, iterations=1)
    
    return Image.fromarray(processed)

def ocr_image(image_base64: str) -> str:
    """
    Perform OCR on a base64-encoded image after preprocessing.
    Returns clean, stripped text.
    """
    try:
        # Decode base64 to bytes
        img_bytes = base64.b64decode(image_base64)
        
        # Preprocess the image
        preprocessed_img = preprocess_image_for_ocr(img_bytes)
        
        # OCR
        text = pytesseract.image_to_string(preprocessed_img, lang='eng')
        
        # Clean up excessive whitespace
        clean_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        return clean_text
    
    except Exception as e:
        return f"OCR error: {str(e)}"
