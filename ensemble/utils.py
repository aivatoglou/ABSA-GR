import re
import unicodedata


def language_model_preprocessing(text):

    # Remove url
    text = re.sub(r"(\w+:\/\/\S+)", " ", text)
    # Remove dates and time
    text = re.sub(r"[0-9]{2}[\/,:][0-9]{2}[\/,:][0-9]{2,4}", " ", text)
    # Remove emails
    text = re.sub(r"(\w+@\w+.[\w+]{2,4})", " ", text)
    # Remove percentages
    text = re.sub(r"\d+(\%|\s\bpercent\b)", " ", text)
    # Remove special symbols
    text = re.sub(r"#|@|:|-", " ", text)

    if text.startswith("RT") | text.startswith("rt"):
        text = text[len("RT") :]

    # Multiline to single line codes
    text = " ".join(text.splitlines())

    # Remove extra spaces
    text = re.sub(" +", " ", text)
    # Strip string
    text = text.strip()

    # Needed in order to find if aspect is in text
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

    return text


def translated_preprocessing(text):

    # Remove digits
    text = re.sub(r"[~^0-9]", " ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove duplicated words in row
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)

    # Remove url
    text = re.sub(r"(\w+:\/\/\S+)", " ", text)
    # Remove dates and time
    text = re.sub(r"[0-9]{2}[\/,:][0-9]{2}[\/,:][0-9]{2,4}", " ", text)
    # Remove emails
    text = re.sub(r"(\w+@\w+.[\w+]{2,4})", " ", text)
    # Remove percentages
    text = re.sub(r"\d+(\%|\s\bpercent\b)", " ", text)
    # Remove special symbols
    text = re.sub(r"#|@|:|-", " ", text)

    if text.startswith("RT") | text.startswith("rt"):
        text = text[len("RT") :]

    # Multiline to single line codes
    text = " ".join(text.splitlines())

    # Remove extra spaces
    text = re.sub(" +", " ", text)
    # Strip string
    text = text.strip()

    # Needed in order to find if aspect is in text
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

    return text
