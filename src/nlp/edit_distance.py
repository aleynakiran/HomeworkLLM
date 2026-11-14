"""Correct single-word product names using Levenshtein distance."""
import re

PRODUCTS = ('dishwasher', 'refrigerator', 'oven', 'dryer', 'buzdolabı', 'fırın', 'kurutucu')


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def correct_products(text):
    def correct(match):
        word = match.group()
        best = min(PRODUCTS, key=lambda product: distance(word, product))
        return best if len(word) >= 4 and distance(word, best) <= 2 else word
    return re.sub(r'[^\W\d_]+', correct, text, flags=re.UNICODE)
