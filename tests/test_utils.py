import pytest
from src.utils import detect_city

@pytest.mark.parametrize("text, expected", [
    ("How do I get from the airport to Lisboa?", "Lisbon"),
    ("Best beer halls in München", "Munich"),
    ("Is Munich more expensive than Zurich?", None),         # two cities
    ("How do I get from the airport to the centre?", None),  # no city
    ("Things to do in Milano", "Milan"),
    ("Where can I eat pintxos in Bilbo?", "Bilbao"),
])
def test_detect_city(text, expected):
    assert detect_city(text) == expected