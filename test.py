import os
from dotenv import load_dotenv

load_dotenv()

print("OpenRouter key found:", bool(os.getenv("OPENROUTER_API_KEY")))