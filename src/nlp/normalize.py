"""Simple Turkish-aware case folding and repeated-letter normalization."""
import re


def normalize(text):
    text = text.replace('I', 'ı').replace('İ', 'i').lower()
    return re.sub(r'(\w)\1{2,}', r'\1\1', text).strip()
