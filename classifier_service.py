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
        
        logger.info(f"Request payload: {payload}")
        
        # Use synchronous requests for now
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=30  # Increase timeout
        )
        
        if response.status_code != 200:
            logger.error(f"HF API returned status code {response.status_code}: {response.text}")
            response.raise_for_status()
            
        result = response.json()
        logger.info(f"API response: {result}")
        
        # Handle the response according to the API's actual response format
        # Assuming the response has a similar structure to the example
        try:
            # Get the highest scoring label
            top_label_index = 0
            if 'scores' in result and len(result['scores']) > 0:
                top_label_index = result['scores'].index(max(result['scores']))
                
            return {
                "label": result['labels'][top_label_index] if 'labels' in result else DOCUMENT_TYPES[0],
                "confidence": round(result['scores'][top_label_index], 4) if 'scores' in result else 0.0
            }
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse API response: {str(e)}")
            logger.error(f"Response content: {result}")
            raise RuntimeError(f"Failed to parse API response: {str(e)}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"HF API request failed: {str(e)}")
        raise RuntimeError(f"HF API request failed: {str(e)}")