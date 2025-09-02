#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# utils/gcs.py
import os
from google.cloud import storage
from datetime import timedelta

def _client() -> storage.Client:
    """
    Create a Google Cloud Storage client.
    Relies on:
      - GOOGLE_APPLICATION_CREDENTIALS (path to service account JSON)
      - GCP_PROJECT (your GCP project id)
    Both should be set in your .env file.
    """
    project_id = os.getenv("GCP_PROJECT")
    return storage.Client(project=project_id)

def signed_url(bucket: str, blob_path: str, minutes: int = 15) -> str:
    """
    Generate a signed URL for a GCS object.
    """
    client = _client()
    blob = client.bucket(bucket).blob(blob_path)
    return blob.generate_signed_url(
        expiration=timedelta(minutes=minutes),
        method="GET",
    )

