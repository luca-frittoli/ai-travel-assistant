import json
import re
from src.config import CITIES
from src.utils import chunk_words

guides_path = 'guides/'
url = "https://en.wikivoyage.org/wiki/"
HEADING = re.compile(r"(={2,})\s*(.+?)\s*\1")

if __name__ == '__main__':
    for city in CITIES:
        with open(guides_path + city + '.json', 'r', encoding='utf-8') as f:
            guide = json.load(f)["query"]["pages"][0]["extract"]

        chunk_list = chunk_words(
            extract=guide
        )

        carried_section = "Intro"

        chunks = []

        for i, chunk_str in enumerate(chunk_list):
            chunk = {
                "id": f"{city}-{i:03d}",
                "city": city,
                "path": [carried_section],
                "sections": [carried_section],
                "text": None,
                "url": url + city.replace(" ", "_")
            }
            for section_found in HEADING.finditer(chunk_str):

                if len(section_found.group(1)) == 2:
                    carried_section = section_found.group(2)
                    chunk["sections"].append(carried_section)
            chunk["sections"] = list(dict.fromkeys(chunk["sections"]))

            text = HEADING.sub(" ", chunk_str)  # remove the "== Title ==" markers
            text = " ".join(text.split())  # collapse the leftover double spaces
            chunk["text"] = f"{city}\n{text}"  # add the city name at the start

            chunks.append(chunk)

        with open(guides_path + city + '_chunked_fixed.json', 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
