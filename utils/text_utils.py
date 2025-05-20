import re, string

# Utils from https://github.com/princeton-nlp/ALCE/blob/246c476a4edfc564266b7346b6e29ef4861ae937/utils.py


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_presence(answers, context):
    """Verify if any of the answers is present in the given context."""

    answers = [normalize_answer(ans) for ans in answers]
    context = normalize_answer(context)

    for ans in answers:
        if ans in context:
            return True

    return False


def clean_for_ner(text):
    # Remove bullet points (replace with newline for better segmentation)
    text = re.sub(r"•\s*", "\n", text)

    # Remove hyphen-based list markers
    text = re.sub(r"\n-\s*", "\n", text)

    # Normalize quotes that may confuse tokenization
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)

    # Remove nested quotes (e.g., "Babyface")
    text = re.sub(r'["\']([^"\']+)["\']', r"\1", text)

    # Normalize spacing and strip each line
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    text = "\n".join(lines)

    # Ensure it ends with punctuation
    if not text.endswith((".", "!", "?")):
        text += "."

    return text
