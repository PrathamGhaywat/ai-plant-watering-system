#183086

import requests

API_KEY = "WS7azRDQvNhDO1RhmAIl16H_AI0vnI3Wo38iRp7xBeY"
latin_name = "Peperomia pellucida (L.) Kunth"

search_url = f"http://trefle.io/api/v1/plants/183086?token={API_KEY}"
response = requests.get(search_url)

if response.status_code == 200:
    results = response.json()["data"]
    if results:
        # Take the first result
        plant = results
        plant_id = plant["id"]
        print("Found:", plant["scientific_name"], "(ID:", plant_id, ")")
    else:
        print("No matches found for", latin_name)
else:
    print("Error:", response.status_code, response.text)