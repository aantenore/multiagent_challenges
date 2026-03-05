import os
import dotenv
from google import genai

dotenv.load_dotenv(override=True)
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
models = [m.name for m in client.models.list() if 'gemini' in m.name.lower()]
with open('gemini_models_output.txt', 'w') as f:
    f.write("AVAILABLE GEMINI Models:\n")
    for m in models:
        f.write(m + "\n")
