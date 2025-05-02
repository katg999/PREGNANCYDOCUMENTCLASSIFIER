import boto3
import os
from botocore.exceptions import ClientError

def get_spaces_client():
    return boto3.client(
        's3',
        endpoint_url=os.getenv('SPACES_ENDPOINT'),
        aws_access_key_id=os.getenv('DO_SPACES_KEY'),
        aws_secret_access_key=os.getenv('DO_SPACES_SECRET')
    )

async def store_document(content: bytes, patient_id: str, doc_type: str, filename: str) -> str:
    """Store file in DigitalOcean Spaces with organized structure"""
    client = get_spaces_client()
    safe_type = doc_type.replace(" ", "_").lower()
    key = f"patients/{patient_id}/{safe_type}/{filename}"
    
    try:
        client.put_object(
            Bucket=os.getenv('BUCKET_NAME'),
            Key=key,
            Body=content,
            ACL='private'  # Change to 'public-read' if needed
        )
        return f"{os.getenv('SPACES_ENDPOINT')}/{os.getenv('BUCKET_NAME')}/{key}"
    except ClientError as e:
        raise RuntimeError(f"Spaces upload failed: {str(e)}")