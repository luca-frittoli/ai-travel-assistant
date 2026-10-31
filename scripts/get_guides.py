import json
import requests
from src.config import CITIES

url = "https://en.wikivoyage.org/w/api.php"
headers = {"User-Agent": "ai-travel-assistant learning project my@email.com"}
params = {
    "action": "query",
    "format": "json",
    "prop": "extracts",
    "explaintext": 1,
    "redirects": 1,
    "formatversion": 2,
}
out_path = "guides/"

if __name__ == '__main__':
    for city in CITIES:
        print(f"Getting {city} guide...")
        tmp_params = params.copy()
        tmp_params["titles"] = city
        guide = requests.get(url=url, headers=headers, params=tmp_params).json()

        with open(out_path + city + '.json', 'w') as f:
            json.dump(guide, f, ensure_ascii=False, indent=2)

    print(f"All guides saved successfully under {out_path}")