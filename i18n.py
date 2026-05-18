import json
import os

from config import DEFAULT_LANG, SUPPORTED_LANGS

LOCALES_DIR = "locales"


def load_locale(lang: str) -> dict:
    path = os.path.join(LOCALES_DIR, f"{lang}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_translations(lang: str) -> dict:
    base = load_locale(DEFAULT_LANG)
    selected = load_locale(lang)
    merged = base.copy()
    merged.update(selected)
    return merged


def get_supported_language_labels() -> dict:
    labels = {}
    for code in SUPPORTED_LANGS:
        locale = load_locale(code)
        labels[code] = locale.get("lang_name", code)
    return labels
