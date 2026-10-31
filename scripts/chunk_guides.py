import json
import re
from src.config import CITIES

guides_path = 'guides/'
url = "https://en.wikivoyage.org/wiki/"
HEADING = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$")

if __name__ == '__main__':
    for city in CITIES:
        with open(guides_path + city + '.json', 'r', encoding='utf-8') as f:
            guide = json.load(f)["query"]["pages"][0]["extract"]

        lines = guide.splitlines()

        chunks = [
            {
                "id": f"{city}-000",
                "city": city,
                "path": ["Intro"],
                "section": "Intro",
                "text": f"{city}\n",
                "url": url + city
            }
        ]

        sec_id = 0

        previous_elem = chunks[-1]

        for line in lines:
            sec = HEADING.match(line)
            if sec is not None:
                sec_id += 1
                level = len(sec.group(1)) - 2
                path = previous_elem["path"][:level] + [sec.group(2)]
                section = path[0]
                new_elem = {
                    "id": f"{city}-{sec_id:03d}",
                    "city": city,
                    "path": path,
                    "section": section,
                    "text": f"{'>'.join([city] + path)}\n",
                    "url": url + city.replace(" ", "_") + '#' + path[-1].replace(" ", "_")
                }
                chunks.append(new_elem)
                previous_elem = new_elem
            else:
                previous_elem["text"] += '\n' + line

        for chunk in chunks:
            chunk["text"] = chunk["text"].strip()

        chunks = [chunk for chunk in chunks if "\n" in chunk["text"]]

        with open(guides_path + city + '_chunked.json', 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
