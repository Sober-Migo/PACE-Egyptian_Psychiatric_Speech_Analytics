# Environment and secure configuration mapping
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables securely from .env
load_dotenv()

class Settings:
    HF_TOKEN = os.getenv("HF_TOKEN")

settings = Settings()

def authenticate_huggingface():
    """Authenticates the environment with Hugging Face for private model pulls."""
    if settings.HF_TOKEN:
        try:
            login(token=settings.HF_TOKEN)
            print("✅ Successfully Authenticated with Hugging Face.")
        except Exception as e:
            print(f"🚨 HF Authentication rejected: {e}")
    else:
        print("⚠️ Warning: No HF_TOKEN found in .env file. Relying on public access.")