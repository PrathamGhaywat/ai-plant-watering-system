#1376726

import requests

API_KEY = "WS7azRDQvNhDO1RhmAIl16H_AI0vnI3Wo38iRp7xBeY"
plant_id = 1376726  # Example species ID for Bitter Lettuce

url = f"https://trefle.io/api/v1/plants/{plant_id}?token={API_KEY}"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print("Plant:", data["data"]["common_name"])
    print("Scientific:", data["data"]["scientific_name"])
    print("Growth info:", data["data"]["main_species"]["growth"])
else:
    print("Error:", response.status_code, response.text)
