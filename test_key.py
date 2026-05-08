from dotenv import load_dotenv
from pathlib import Path
import os

# Force load from exact path
env_path = Path("C:/Users/hp/OneDrive/Desktop/DocuMind/.env")
print("ENV file exists:", env_path.exists())
print("ENV file path:", env_path)

load_dotenv(dotenv_path=env_path, override=True)

key = os.getenv("GEMINI_API_KEY", "")
print("Key found:", bool(key))
print("Key length:", len(key))
print("Key value:", repr(key))
print("Has quotes:", key.startswith('"'))
print("Has spaces:", " " in key)