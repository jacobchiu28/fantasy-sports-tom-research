import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure with your API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

print("Available Google Generative AI models:")
print("=" * 50)

try:
    models = genai.list_models()
    for model in models:
        # Check if it supports generateContent
        if 'generateContent' in model.supported_generation_methods:
            print(f"✅ {model.name} - Supports generateContent")
        else:
            print(f"❌ {model.name} - Does NOT support generateContent")
except Exception as e:
    print(f"Error listing models: {e}")
