"""Find appliance-style model codes without changing the message."""
import re


def extract_model_code(text):
    match = re.search(r'\b[A-Za-z]{2,5}[- ]\d{3,6}\b', text)
    return match.group().upper().replace(' ', '-') if match else None
