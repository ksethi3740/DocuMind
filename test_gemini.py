from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path("C:/Users/hp/OneDrive/Desktop/DocuMind/.env"), override=True)

import os
key = os.getenv("GEMINI_API_KEY", "").strip()
print("Key:", repr(key))

from google import genai
client = genai.Client(api_key=key)
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Say hello and confirm you are working."
)
print("Gemini says:", response.text)