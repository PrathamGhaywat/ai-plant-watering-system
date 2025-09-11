#google/gemini-2.0-flash-exp:free

import requests
import json
import base64
from pathlib import Path
API_KEY_REF = "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"
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ai(prompt, img_files):
    """
    Use the sonoma dusk model to prompt stuff. requires an image
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY_REF}",
        "Content-Type": "application/json"
    }

    # Read and encode the image
    image_path = img_files
    base64_image = encode_image_to_base64(image_path)
    data_url = f"data:image/jpeg;base64,{base64_image}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
        }
    ]

    payload = {
        "model": "openrouter/sonoma-dusk-alpha",
        "messages": messages
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]