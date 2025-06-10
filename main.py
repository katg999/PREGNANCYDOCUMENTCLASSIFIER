from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import io
import os
import logging
from classifier_service import classify_document
from spaces_service import store_document
from tenacity import RetryError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Pregnancy Document Classifier")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Pregnancy Classifier API - Up and running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Pregnancy Document Classifier"}

@app.get("/test-huggingface")
async def test_huggingface():
    """Test endpoint to verify Hugging Face API connection"""
    try:
        # Check environment variables
        if not os.getenv("MODEL_ENDPOINT"):
            return {"status": "error", "message": "MODEL_ENDPOINT environment variable is not set"}
        
        if not os.getenv("HF_API_TOKEN"):
            return {"status": "error", "message": "HF_API_TOKEN environment variable is not set"}
        
        # Use a simple test text
        test_text = "This is a test for ultrasound report classification."
        
        # Try to classify
        logger.info("Testing Hugging Face API connection...")
        try:
            classification = await classify_document(test_text)
            return {
                "status": "success",
                "message": "Connection to Hugging Face API successful",
                "classification": classification,
                "model_endpoint": os.getenv("MODEL_ENDPOINT")
            }
        except RetryError as e:
            # Handle retry error specifically
            return {
                "status": "error",
                "message": "Hugging Face API is currently unavailable after multiple retry attempts",
                "model_endpoint": os.getenv("MODEL_ENDPOINT"),
                "error_details": str(e)
            }
        
    except Exception as e:
        logger.error(f"Hugging Face API test failed: {str(e)}")
        return {
            "status": "error", 
            "message": f"Connection to Hugging Face API failed: {str(e)}",
            "model_endpoint": os.getenv("MODEL_ENDPOINT")
        }

@app.post("/classify")
async def classify_endpoint(
    file: UploadFile = File(...),
    patient_id: str = Form(...)
):
    try:
        # 1. Validate file type
        if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            raise HTTPException(400, detail="Invalid file type. Only PDF/JPEG/PNG allowed")
        
        # 2. Read and extract text
        contents = await file.read()
        extracted_text = extract_text(io.BytesIO(contents), file.filename)
        
        # === ADD DEBUG LOGGING HERE ===
        logger.info(f"Extracted text sample: {extracted_text[:500]}...")  # Log first 500 chars
        logger.info(f"Sending this text to classifier: {extracted_text}")
        # ==============================
        
        # 3. Classify
        try:
            classification = await classify_document(extracted_text)
            
            # === ADD LOGGING FOR CLASSIFICATION RESULT ===
            logger.info(f"Raw classification response: {classification}")
            # ============================================
            
        except RetryError:
            # If Hugging Face API is unavailable, use a fallback classification
            logger.warning("Hugging Face API unavailable, using fallback classification")
            classification = {
                "label": "unclassified document",  # Fallback label
                "confidence": 0.0,
                "status": "fallback_used"
            }
        
        # 4. Store
        s3_path = await store_document(contents, patient_id, 
                                      classification.get('label', 'unclassified'), 
                                      file.filename)
        
        return {
            "patient_id": patient_id,
            "classification": classification,
            "s3_path": s3_path,
            "status": "processed"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(500, detail=f"Document processing failed: {str(e)}")

def extract_text(file_stream, filename: str) -> str:
    """Extract text from PDF or image using Tesseract OCR"""
    try:
        if filename.lower().endswith('.pdf'):
            images = convert_from_bytes(file_stream.read())
            text = "\n".join(pytesseract.image_to_string(img) for img in images)
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
        else:
            text = pytesseract.image_to_string(Image.open(file_stream))
            logger.info(f"Extracted {len(text)} characters from image")
            return text
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        raise RuntimeError(f"Text extraction failed: {str(e)}")