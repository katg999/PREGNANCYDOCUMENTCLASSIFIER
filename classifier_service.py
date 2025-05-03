import os
import requests
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, before_log, after_log

# Configure logging
logger = logging.getLogger(__name__)

HF_API_URL = os.getenv("MODEL_ENDPOINT")
DOCUMENT_TYPES = [
    "ultrasound report",
    "blood test results",
    "urine analysis",
    "prenatal screening"
]

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before=before_log(logger, logging.INFO),
    after=after_log(logger, logging.INFO)
)
async def classify_document(text: str) -> dict:
    """Classify document using Hugging Face API"""
    if not HF_API_URL:
        logger.error("MODEL_ENDPOINT environment variable is not set")
        raise RuntimeError("HF API URL not configured")
        
    token = os.getenv('HF_API_TOKEN')
    if not token:
        logger.error("HF_API_TOKEN environment variable is not set")
        raise RuntimeError("HF API token not configured")
        
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        logger.info(f"Sending request to Hugging Face API at {HF_API_URL}")
        logger.info(f"Text length: {len(text)} characters")
        
        # Use synchronous requests library within async function
        # This is not ideal but works with the current structure
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={
                "inputs": text[:5000],  # Limit text size for API call
                "parameters": {"candidate_labels": DOCUMENT_TYPES}
            },
            timeout=30  # Increase timeout
        )
        
        if response.status_code != 200:
            logger.error(f"HF API returned status code {response.status_code}: {response.text}")
            response.raise_for_status()
            
        result = response.json()
        logger.info(f"Successfully received classification: {result['labels'][0]}")
        
        return {
            "label": result['labels'][0],
            "confidence": round(result['scores'][0], 4)
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"HF API request failed: {str(e)}")
        raise RuntimeError(f"HF API request failed: {str(e)}")