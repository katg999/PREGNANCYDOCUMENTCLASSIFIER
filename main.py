from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import io
import logging
from classifier_service import classify_document
from spaces_service import store_document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Pregnancy Document Classifier")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Pregnancy Classifier API - Up and running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Pregnancy Document Classifier"}

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

        # 3. Classify
        classification = await classify_document(extracted_text)

        # 4. Store
        s3_path = await store_document(contents, patient_id, classification['label'], file.filename)

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
        raise HTTPException(500, detail="Document processing failed")

def extract_text(file_stream, filename: str) -> str:
    """Extract text from PDF or image using Tesseract OCR"""
    try:
        if filename.lower().endswith('.pdf'):
            images = convert_from_bytes(file_stream.read())
            return "\n".join(pytesseract.image_to_string(img) for img in images)
        else:
            return pytesseract.image_to_string(Image.open(file_stream))
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        raise RuntimeError("Text extraction failed")
