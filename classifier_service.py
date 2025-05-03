import os
import requests
import logging
import time
from tenacity import retry, retry_if, stop_after_attempt, wait_exponential, before_log, after_log, retry_if_exception_type

# Configure logging
logger = logging.getLogger(__name__)

HF_API_URL = os.getenv("MODEL_ENDPOINT")
DOCUMENT_TYPES = [
    "ultrasound report",
    "blood test results",
    "urine analysis",
    "prenatal screening"
]

# Custom retry strategy: Don't retry on 503 Service Unavailable
def should_retry_exception(exception):
    if isinstance(exception, requests.exceptions.RequestException):
        # If it's a 503 error, don't retry
        if hasattr(exception, 'response') and exception.response is not None:
            if exception.response.status_code == 503:
                logger.warning("Received 503 Service Unavailable, not retrying")
                return False
    return True

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before=before_log(logger, logging.INFO),
    after=after_log(logger, logging.INFO),
    retry=retry_if(lambda e: isinstance(e, requests.exceptions.RequestException) and should_retry_exception(e))
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
        
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        logger.info(f"Sending request to Hugging Face API at {HF_API_URL}")
        logger.info(f"Text length: {len(text)} characters")
        
        # Limit text size to prevent issues with large documents
        limited_text = text[:5000] if len(text) > 5000 else text
        
        # Format the payload exactly as shown in the example
        payload = {
            "inputs": limited_text,
            "parameters": {
                "candidate_labels": ", ".join(DOCUMENT_TYPES)
            }
        }
        
        # Use synchronous requests for now
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=30  # Increase timeout
        )
        
        if response.status_code == 503:
            logger.error(f"HF API returned status code 503: {response.text}")
            # For 503 errors, directly raise to avoid retries
            raise RuntimeError(f"Hugging Face API is currently unavailable (503): {response.text}")
        elif response.status_code != 200:
            logger.error(f"HF API returned status code {response.status_code}: {response.text}")
            response.raise_for_status()
            
        result = response.json()
        logger.info(f"API response received successfully")
        
        # Handle the response according to the API's actual response format
        try:
            # Get the highest scoring label
            top_label_index = 0
            if 'scores' in result and len(result['scores']) > 0:
                top_label_index = result['scores'].index(max(result['scores']))
                
            return {
                "label": result['labels'][top_label_index] if 'labels' in result else DOCUMENT_TYPES[0],
                "confidence": round(result['scores'][top_label_index], 4) if 'scores' in result else 0.0,
                "status": "success"
            }
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse API response: {str(e)}")
            logger.error(f"Response content: {result}")
            return {
                "label": "unknown document", 
                "confidence": 0.0,
                "status": "parse_error",
                "error": str(e)
            }
            
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 503:
            logger.error(f"HF API unavailable (503): {str(e)}")
            # Don't retry 503 errors
            raise RuntimeError(f"Hugging Face API is currently unavailable (503): {str(e)}")
        else:
            logger.error(f"HF API request failed: {str(e)}")
            raise