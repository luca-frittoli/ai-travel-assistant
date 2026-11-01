import json
import re
from src.config import CITIES, CHUNK_SIZE, SHORT_TAIL_TOL
from src.utils import chunk_words

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
                "sections": ["Intro"],
                "text": f"{city}\n",
                "url": url + city.replace(" ", "_")
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
                    "sections": [section],
                    "text": f"{'>'.join([city] + path)}\n",
                    "url": url + city.replace(" ", "_") + '#' + path[-1].replace(" ", "_")
                }
                chunks.append(new_elem)
                previous_elem = new_elem
            else:
                previous_elem["text"] += '\n' + line

        final_chunks = []
        for chunk in chunks:
            breadcrumb, text = chunk["text"].split('\n', 1)
            if len(text.split()) > CHUNK_SIZE+SHORT_TAIL_TOL:
                chunked_text = chunk_words(text)
                for j, sub_text in enumerate(chunked_text):
                    new_elem = {
                        "id": f"{chunk['id']}-{j}",
                        "city": city,
                        "path": chunk["path"],
                        "sections": chunk["sections"],
                        "text": f"{breadcrumb}\n{sub_text}",
                        "url": chunk["url"]
                    }
                    final_chunks.append(new_elem)
            else:
                final_chunks.append(chunk)

        for chunk in final_chunks:
            chunk["text"] = chunk["text"].strip()

        chunks = [chunk for chunk in final_chunks if "\n" in chunk["text"]]

        with open(guides_path + city + '_chunked_hybrid.json', 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
