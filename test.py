import requests
import os

API_URL = "https://api.openai.com/v1/chat/completions"
API_KEY = os.getenv("GPT_API_KEY")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Categorize the following text based on criteria: example text."}
    ],
    "max_tokens": 500
}

response = requests.post(API_URL, headers=headers, json=data)
print(response.json())
