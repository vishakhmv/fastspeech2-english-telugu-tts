import re
import unicodedata

VIRAMA = "్"
ANUSVARA = {"ం", "ఁ"}
VISARGA = "ః"

CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "c", "ఛ": "c", "జ": "j", "ఝ": "j", "ఞ": "ny",
    "ట": "T", "ఠ": "Th", "డ": "D", "ఢ": "D", "ణ": "N",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "r", "ల": "l", "ళ": "L", "వ": "v",
    "శ": "sh", "ష": "S", "స": "s", "హ": "h",
}

INDEPENDENT_VOWELS = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"],
    "ఋ": ["r", "u"], "ౠ": ["r", "uu"], "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"],
    "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}

VOWEL_SIGNS = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"],
    "ృ": ["r", "u"], "ౄ": ["r", "uu"], "ె": ["e"], "ే": ["ee"], "ై": ["ai"],
    "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"],
}

VELAR = set("కఖగఘఙ")
PALATAL = set("చఛజఝఞ")
RETROFLEX = set("టఠడఢణ")
DENTAL = set("తథదధన")

VOWEL_PHONES = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "o", "oo", "ai", "au"]
CONSONANT_PHONES = [
    "k", "kh", "g", "gh", "ng",
    "c", "j", "ny",
    "T", "Th", "D", "N",
    "t", "th", "d", "dh", "n",
    "p", "ph", "b", "bh", "m",
    "y", "r", "l", "L", "v", "sh", "S", "s", "h",
]
SILENCE_PHONE = "sp"
PHONES = VOWEL_PHONES + CONSONANT_PHONES

TELUGU_FIRST = 0x0C00
TELUGU_LAST = 0x0C7F
TELUGU_BLOCK = chr(TELUGU_FIRST) + "-" + chr(TELUGU_LAST)
INVISIBLE_CHARACTERS = "".join(chr(code) for code in (0x200B, 0x200C, 0x200D, 0xFEFF))
PUNCTUATION_PAUSE = set(",;:.!?")

DIGITS_TELUGU = {chr(0x0C66 + i): str(i) for i in range(10)}

UNITS = [
    "సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
    "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు",
    "పదిహేడు", "పద్దెనిమిది", "పంతొమ్మిది",
]

TENS = {
    2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై",
    6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై",
}

HUNDRED_ALONE = "వంద"
HUNDRED_COMPOUND = "నూట"
HUNDRED_PLURAL = "వందలు"
HUNDRED_OBLIQUE = "వందల"
THOUSAND_ONE = "వెయ్యి"
THOUSAND_PLURAL = "వేలు"
THOUSAND_OBLIQUE = "వేల"
LAKH_ONE = "లక్ష"
LAKH_PLURAL = "లక్షలు"
LAKH_OBLIQUE = "లక్షల"
CRORE_ONE = "కోటి"
CRORE_PLURAL = "కోట్లు"
CRORE_OBLIQUE = "కోట్ల"
POINT_WORD = "పాయింట్"
PERCENT_WORD = "శాతం"
FIRST_ORDINAL = "మొదటి"
ORDINAL_SUFFIX = "వ"


def below_hundred(n):
    if n < 20:
        return UNITS[n]
    tens, unit = divmod(n, 10)
    word = TENS[tens]
    return word if unit == 0 else word + " " + UNITS[unit]


def number_to_words(n):
    if n == 0:
        return UNITS[0]
    crore, rest = divmod(n, 10 ** 7)
    lakh, rest = divmod(rest, 10 ** 5)
    thousand, rest = divmod(rest, 1000)
    hundred, rest = divmod(rest, 100)
    parts = []
    if crore:
        if crore == 1:
            parts.append(CRORE_ONE)
        else:
            tail = lakh or thousand or hundred or rest
            parts.append(number_to_words(crore) + " " + (CRORE_OBLIQUE if tail else CRORE_PLURAL))
    if lakh:
        if lakh == 1:
            parts.append(LAKH_ONE)
        else:
            tail = thousand or hundred or rest
            parts.append(below_hundred(lakh) + " " + (LAKH_OBLIQUE if tail else LAKH_PLURAL))
    if thousand:
        if thousand == 1:
            parts.append(THOUSAND_ONE)
        else:
            tail = hundred or rest
            parts.append(below_hundred(thousand) + " " + (THOUSAND_OBLIQUE if tail else THOUSAND_PLURAL))
    if hundred:
        if hundred == 1:
            parts.append(HUNDRED_COMPOUND if rest else HUNDRED_ALONE)
        else:
            parts.append(UNITS[hundred] + " " + (HUNDRED_OBLIQUE if rest else HUNDRED_PLURAL))
    if rest:
        parts.append(below_hundred(rest))
    return " ".join(parts)


def ordinal_words(n):
    if n == 1:
        return FIRST_ORDINAL
    words = number_to_words(n).split(" ")
    last = words[-1]
    if last.endswith("ై"):
        last = last[:-1] + "య్య"
    elif last.endswith("ు") or last.endswith("ి"):
        last = last[:-1]
    return " ".join(words[:-1] + [last + ORDINAL_SUFFIX])


def digits_to_ascii(text):
    return "".join(DIGITS_TELUGU.get(c, c) for c in text)


def expand_numbers(text):
    text = digits_to_ascii(text)

    def time_replace(match):
        hours, minutes = int(match.group(1)), int(match.group(2))
        words = number_to_words(hours)
        if minutes:
            words += " " + number_to_words(minutes)
        return " " + words + " "

    text = re.sub(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", time_replace, text)

    def ordinal_replace(match):
        return " " + ordinal_words(int(match.group(1))) + " "

    text = re.sub(r"(\d+)\s*వ(?![" + TELUGU_BLOCK + "])", ordinal_replace, text)

    def decimal_replace(match):
        whole = number_to_words(int(match.group(1)))
        fraction = " ".join(UNITS[int(d)] for d in match.group(2))
        return " " + whole + " " + POINT_WORD + " " + fraction + " "

    text = re.sub(r"(\d+)\.(\d+)", decimal_replace, text)

    def integer_replace(match):
        raw = match.group(0).replace(",", "")
        if len(raw) > 9:
            return " " + " ".join(UNITS[int(d)] for d in raw) + " "
        return " " + number_to_words(int(raw)) + " "

    text = re.sub(r"\d{1,3}(?:,\d{2,3})+|\d+", integer_replace, text)
    return text.replace("%", " " + PERCENT_WORD + " ")


def is_telugu_char(ch):
    return TELUGU_FIRST <= ord(ch) <= TELUGU_LAST


def normalize(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub("[" + INVISIBLE_CHARACTERS + "]", "", text)
    text = "".join(
        "ం" if ch == "o" and i > 0 and is_telugu_char(text[i - 1]) else ch
        for i, ch in enumerate(text)
    )
    text = expand_numbers(text)
    text = text.replace("।", ".").replace("…", ".").replace("—", ",").replace("–", ",")
    text = re.sub(r"\s-\s", ", ", text)
    text = re.sub(r"[\"'“”‘’()\[\]{}<>*_/\\|~^`#@&+=]", " ", text)
    text = re.sub("(?<=[" + TELUGU_BLOCK + "]):(?=[" + TELUGU_BLOCK + "])", ", ", text)
    text = re.sub("[^" + TELUGU_BLOCK + r",;:.!? \-]", " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)
    text = re.sub(r"([,;:.!?])\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def nasal_before(next_char):
    if next_char in VELAR:
        return "ng"
    if next_char in PALATAL:
        return "ny"
    if next_char in RETROFLEX:
        return "N"
    if next_char in DENTAL:
        return "n"
    return "m"


def word_to_phones(word):
    phones = []
    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        if ch in CONSONANTS:
            phones.append(CONSONANTS[ch])
            following = word[i + 1] if i + 1 < n else ""
            if following in VOWEL_SIGNS:
                phones.extend(VOWEL_SIGNS[following])
                i += 2
            elif following == VIRAMA:
                i += 2
            else:
                phones.append("a")
                i += 1
        elif ch in INDEPENDENT_VOWELS:
            phones.extend(INDEPENDENT_VOWELS[ch])
            i += 1
        elif ch in ANUSVARA:
            phones.append(nasal_before(word[i + 1] if i + 1 < n else ""))
            i += 1
        elif ch == VISARGA:
            phones.append("h")
            i += 1
        else:
            i += 1
    return phones


def tokenize(normalized):
    return re.findall("[" + TELUGU_BLOCK + "]+|[,;:.!?]", normalized)


def words_of(text):
    return [t for t in tokenize(normalize(text)) if t not in PUNCTUATION_PAUSE]


def text_to_phones_and_words(text):
    phones = []
    words = []
    for token in tokenize(normalize(text)):
        if token in PUNCTUATION_PAUSE:
            if phones and phones[-1] != SILENCE_PHONE:
                phones.append(SILENCE_PHONE)
        else:
            start = len(phones)
            phones.extend(word_to_phones(token))
            words.append((token, start, len(phones)))
    while phones and phones[-1] == SILENCE_PHONE:
        phones.pop()
    return phones, words


def text_to_phones(text):
    return text_to_phones_and_words(text)[0]


def lexicon_entries(texts):
    entries = {}
    for text in texts:
        for word in words_of(text):
            if word not in entries:
                entries[word] = word_to_phones(word)
    return entries


ISO_15919 = {
    "aa": "ā", "ii": "ī", "uu": "ū", "ee": "ē", "oo": "ō",
    "ng": "ṅ", "ny": "ñ", "T": "ṭ", "Th": "ṭh", "D": "ḍ", "N": "ṇ",
    "L": "ḷ", "sh": "ś", "S": "ṣ",
}


def romanize(phones):
    return "".join(ISO_15919.get(p, p) for p in phones if p != SILENCE_PHONE)
