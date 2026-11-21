"""Alternative names for each city in CITIES, used by detect_city.

Keys match the names in src.config.CITIES exactly (so they match chunk["city"]).
Values are lower-case and include the canonical name itself, so detect_city only
needs this dict. Accented names are listed both with and without accents, so a
plain .lower() on the question is enough; match on word boundaries so that
"milan" doesn't also match inside "milano".

Deliberately left out:
- "Monaco" for Munich: in English it means the country, so it would cause false matches.
  The full Italian name "monaco di baviera" is unambiguous and is included.
"""

CITY_ALTERNATIVE_NAMES: dict[str, list[str]] = {
    "Lisbon": [
        "lisbon",
        "lisboa",        # Portuguese, Spanish
        "lisbona",       # Italian
        "lissabon",      # German, Dutch
        "lisbonne",      # French
    ],
    "Milan": [
        "milan",
        "milano",        # Italian
        "milán",         # Spanish
        "mailand",       # German
    ],
    "Munich": [
        "munich",
        "münchen", "muenchen", "munchen",  # German, with common transliterations
        "monaco di baviera",               # Italian
        "múnich",                          # Spanish
        "munique",                         # Portuguese
    ],
    "Sofia": [
        "sofia",
        "софия",         # Bulgarian
        "sofiya", "sofija",  # transliterations
        "sofía",         # Spanish
    ],
    "Chisinau": [
        "chisinau",
        "chișinău",      # Romanian, comma-below ș (correct form)
        "chişinău",      # cedilla ş (very common in practice)
        "chisinău",
        "kishinev", "kishinyov",  # Russian-derived names
        "кишинёв", "кишинев",     # Russian
    ],
    "Bilbao": [
        "bilbao",
        "bilbo",         # Basque
    ],
    "Zurich": [
        "zurich",
        "zürich", "zuerich",  # German
        "züri",          # Swiss German
        "zurigo",        # Italian
        "zúrich",        # Spanish
    ],
    "Copenhagen": [
        "copenhagen",
        "københavn", "kobenhavn", "koebenhavn",  # Danish
        "kopenhagen",    # German, Dutch
        "copenaghen",    # Italian
        "copenhague",    # Spanish, French, Portuguese
    ],
    "Dublin": [
        "dublin",
        "baile átha cliath", "baile atha cliath",  # Irish
        "dublino",       # Italian
    ],
    "Pamplona": [
        "pamplona",
        "iruña", "iruna",      # Basque / Spanish co-official
        "iruñea", "irunea",    # Basque
    ],
}