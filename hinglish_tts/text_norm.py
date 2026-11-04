"""Text normalization shared by data preparation and inference.

The model only ever sees words, so digits and symbols are spelled out before
tokenization, in Hindi when the sentence is mostly Devanagari and in English
otherwise. The Devanagari danda is mapped to a period because Pocket TTS splits
long inputs into sentences at `. ! ?` only.
"""

import re
import unicodedata

HINDI_0_99 = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ "
    "दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस "
    "बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस "
    "तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस "
    "चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास "
    "पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ "
    "साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर "
    "सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी "
    "अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी "
    "नब्बे इक्यानवे बानवे तिरानवे चौरानवे पचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे"
).split()
assert len(HINDI_0_99) == 100

ENGLISH_0_19 = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
ENGLISH_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()

SYMBOLS = {
    "hi": {"%": " प्रतिशत", "₹": " रुपये", "&": " और ", "+": " प्लस ", "@": " at "},
    "en": {"%": " percent", "₹": " rupees", "&": " and ", "+": " plus ", "@": " at "},
}

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")


def hindi_number(n: int) -> str:
    """Indian numbering: करोड़ (10^7), लाख (10^5), हज़ार (10^3), सौ (10^2)."""
    if n < 100:
        return HINDI_0_99[n]
    parts = []
    for value, word in ((10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        if n >= value:
            parts.append(f"{hindi_number(n // value)} {word}")
            n %= value
    if n:
        parts.append(HINDI_0_99[n])
    return " ".join(parts)


def english_number(n: int) -> str:
    """Indian-English numbering (lakh, crore), as Indian speakers read amounts."""
    if n < 20:
        return ENGLISH_0_19[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return ENGLISH_TENS[tens - 2] + (f" {ENGLISH_0_19[ones]}" if ones else "")
    parts = []
    for value, word in ((10**7, "crore"), (10**5, "lakh"), (1000, "thousand"), (100, "hundred")):
        if n >= value:
            parts.append(f"{english_number(n // value)} {word}")
            n %= value
    if n:
        parts.append(english_number(n))
    return " ".join(parts)


def sentence_language(text: str) -> str:
    """'hi' if Devanagari letters outnumber Latin ones, else 'en'."""
    return "hi" if len(DEVANAGARI.findall(text)) > len(LATIN.findall(text)) else "en"


def _spell_number(match: re.Match, lang: str) -> str:
    digits = match.group(0).replace(",", "")
    speak = hindi_number if lang == "hi" else english_number
    if "." in digits:
        whole, frac = digits.split(".", 1)
        point = " दशमलव " if lang == "hi" else " point "
        return speak(int(whole or 0)) + point + " ".join(speak(int(d)) for d in frac)
    # Long digit strings (phone numbers, OTPs) are read digit by digit.
    if len(digits) > 9 or (len(digits) > 1 and digits.startswith("0")):
        return " ".join(speak(int(d)) for d in digits)
    return speak(int(digits))


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("॥", ".").replace("।", ".")
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    # Devanagari digits -> ASCII so one number speller covers both.
    text = text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    lang = sentence_language(text)
    # "₹500" is read "500 rupees", so move the currency sign after the amount.
    text = re.sub(r"₹\s*(\d[\d,]*(?:\.\d+)?)", r"\1 ₹", text)
    for sym, word in SYMBOLS[lang].items():
        text = text.replace(sym, word)
    text = re.sub(r"\d[\d,]*(?:\.\d+)?", lambda m: _spell_number(m, lang), text)
    text = re.sub(r"\s+([,.?!])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()
