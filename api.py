"""
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="",
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  model="mistralai/mistral-small-3.2-24b-instruct:free",
  messages=[
    {
      "role": "user",
      "content": "Hello"
    }
  ]
)

print(completion.choices[0].message.content)
"""

import requests

API_KEY = "sk-HD7E68bef155d881c12279"

plant_name = "Bitter Lettuce"
url = f"https://perenual.com/api/v2/species-list?key={API_KEY}&q={plant_name}"

response = requests.get(url)

if response.status_code == 200:
    results = response.json()
    if results.get("data"):
        plant = results["data"][0]  # Take the first match
        print("Name:", plant["common_name"])
        print("Scientific:", plant["scientific_name"])
        print("Watering:", plant["watering"])
        print("Sunlight:", plant["sunlight"])
    else:
        print("No results found for", plant_name)
else:
    print("Error:", response.status_code, response.text)

