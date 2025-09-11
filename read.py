import json 

var1 = json.load(open("./metada_files/metada_files/class_idx_to_species_id.json"))
species_c= var1['464']
print(species_c)
var2 = json.load(open("./metada_files/metada_files/plantnet300K_species_id_2_name.json"))
print(var2[species_c])
"""
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  model="deepseek/deepseek-chat-v3.1:free",
  messages=[
    {
      "role": "user",
      "content": f"I have an {var3}. Please give me only the number of Water consumption in milliliter and no other details or text. Please check also the "
    }
  ]
)

print(completion.choices[0].message.content)

import requests
import json

response = requests.post(
  "https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer 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",
    "Content-Type": "application/json",
  },

  json={
    "model": "deepseek/deepseek-chat-v3.1:free",
    "messages": [
      {"role": "user", "content": f"I have an {var3}. Please give me the water consumption in ml"},
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
          "name": "water consumption in ml",
          "strict": True,
          "schema": {
              "type": "object",
              "properties": {
                  "water consumption": {
                      "type": "string",
                      "description": "only Water consumption in ml "
                  }
              }
          }
      },
    },
  }
)
data = response.json()
weather_info = data["choices"][0]["message"]["content"]
"""