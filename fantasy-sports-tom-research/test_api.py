import os
from dotenv import load_dotenv

load_dotenv()

def test_apis():
    # Test OpenAI
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": "Hello, respond with just 'API working'"}],
            max_tokens=10
        )
        print("OpenAI API working")
    except Exception as e:
        print(f"❌ OpenAI API failed: {e}")

    # Test Anthropic
    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello, respond with just 'API working'"}]
        )
        print("Anthropic API working")
    except Exception as e:
        print(f"❌ Anthropic API failed: {e}")

    # Test Google
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('models/gemini-2.5-pro')
        response = model.generate_content("Hello, respond with just 'API working'")
        print("Google API working")
    except Exception as e:
        print(f"❌ Google API failed: {e}")

if __name__ == "__main__":
    test_apis()
