"""Correct product words and phrases using Levenshtein distance."""
import re

PRODUCTS = ('dishwasher', 'refrigerator', 'oven', 'dryer', 'buzdolabı', 'fırın', 'kurutucu')
PHRASES = (('bulaşık', 'makinesi'), ('çamaşır', 'makinesi'))


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def correct_products(text):
    words = list(re.finditer(r'[^\W\d_]+', text, flags=re.UNICODE))
    parts, cursor, i = [], 0, 0

    def close(word, product):
        # Short words need a closer match to avoid changing unrelated words.
        return len(word) >= 4 and distance(word, product) <= (1 if len(word) <= 6 else 2)

    while i < len(words):
        word = words[i]
        parts.append(text[cursor:word.start()])
        # Match adjacent word pairs first, so both words can be corrected together.
        if i + 1 < len(words) and text[word.end():words[i + 1].start()].isspace():
            following = words[i + 1]
            candidates = [phrase for phrase in PHRASES
                          if close(word.group(), phrase[0]) and close(following.group(), phrase[1])]
            if candidates:
                best = min(candidates, key=lambda p: distance(word.group(), p[0]) + distance(following.group(), p[1]))
                parts.append(best[0] + text[word.end():following.start()] + best[1])
                cursor = following.end()
                i += 2
                continue
        candidates = [product for product in PRODUCTS if close(word.group(), product)]
        parts.append(min(candidates, key=lambda p: distance(word.group(), p)) if candidates else word.group())
        cursor = word.end()
        i += 1
    return ''.join(parts) + text[cursor:]
