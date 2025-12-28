import re

# Türkçe sesli harfler
VOWELS = "aeıioöuüAEIİOÖUÜ"

def to_lowercase(text: str) -> str:
    """
    Metni tamamen küçük harfe çevirir
    """
    return text.lower()


def remove_punctuation(text: str) -> str:
    """
    Noktalama işaretlerini siler
    """
    return re.sub(r"[^\w\s]", "", text)


def remove_vowels(text: str) -> str:
    """
    Sesli harfleri siler (deneysel işlem)
    """
    return "".join([c for c in text if c not in VOWELS])


def clean_extra_spaces(text: str) -> str:
    """
    Fazla boşlukları temizler
    """
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_punc: bool = True,
    remove_vowel: bool = False
) -> str:
    """
    Metni seçilen ön-işleme adımlarına göre temizler
    """

    if lowercase:
        text = to_lowercase(text)

    if remove_punc:
        text = remove_punctuation(text)

    if remove_vowel:
        text = remove_vowels(text)

    text = clean_extra_spaces(text)

    return text
