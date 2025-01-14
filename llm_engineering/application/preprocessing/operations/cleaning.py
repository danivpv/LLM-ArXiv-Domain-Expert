import re


def clean_text(text: str) -> str:
    # Replace any character that is not a word character (alphanumeric & underscore), whitespace, or one of the specified punctuation marks (.,!?)
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    # Replace multiple consecutive whitespace characters with a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()
