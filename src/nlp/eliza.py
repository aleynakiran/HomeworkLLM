"""Tiny rule-based greeting and thanks responder."""
import re

RULES = [
    (r'^(hello|hi|hey|merhaba|selam)[!. ]*$', 'Hello! How can I help? / Merhaba! Nasıl yardımcı olabilirim?'),
    (r'^(thanks|thank you|teşekkürler|teşekkür ederim)[!. ]*$', 'You are welcome! / Rica ederim!'),
]


def reply(text):
    for pattern, response in RULES:
        if re.fullmatch(pattern, text):
            return response
    return None
