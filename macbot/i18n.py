"""Bilingual messages for the web pages. The page shows ``en`` or ``zh`` depending on its language switch."""


def tr(en: str, zh: str) -> dict:
    return {"en": en, "zh": zh}


def text(m, lang: str = "en") -> str:
    """Plain text of a message that may be a str or a ``tr`` dict."""
    return m.get(lang) or m.get("en", "") if isinstance(m, dict) else str(m)
