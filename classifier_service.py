import os
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

HF_API_URL = "https://h5juq0gnjnavay71.us-east-1.aws.endpoints.huggingface.cloud"
DOCUMENT_TYPES = [
    "ultrasound report",
    "blood test results",
    "urine analysis",
    "prenatal screening"
]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def classify_document(text: str) -> dict:
    """Classify document using Hugging Face API"""
    headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={
                "inputs": text,
                "parameters": {"candidate_labels": DOCUMENT_TYPES}
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        return {
            "label": result['labels'][0],
            "confidence": round(result['scores'][0], 4)
        }
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HF API request failed: {str(e)}")