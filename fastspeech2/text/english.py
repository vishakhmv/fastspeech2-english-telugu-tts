import re
from pathlib import Path

LEXICON_PATH = Path(__file__).resolve().parents[1] / "lexicon" / "english_lexicon.txt"

ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SCALES = [(10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")]
ORDINAL_IRREGULAR = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}
LETTER_WORD_PREFIX = "ltr"
LETTER_PHONES = {
    "a": ["EY1"], "b": ["B", "IY1"], "c": ["S", "IY1"], "d": ["D", "IY1"], "e": ["IY1"],
    "f": ["EH1", "F"], "g": ["JH", "IY1"], "h": ["EY1", "CH"], "i": ["AY1"],
    "j": ["JH", "EY1"], "k": ["K", "EY1"], "l": ["EH1", "L"], "m": ["EH1", "M"],
    "n": ["EH1", "N"], "o": ["OW1"], "p": ["P", "IY1"], "q": ["K", "Y", "UW1"],
    "r": ["AA1", "R"], "s": ["EH1", "S"], "t": ["T", "IY1"], "u": ["Y", "UW1"],
    "v": ["V", "IY1"], "w": ["D", "AH1", "B", "AH0", "L", "Y", "UW0"],
    "x": ["EH1", "K", "S"], "y": ["W", "AY1"], "z": ["Z", "IY1"],
}
LETTER_WORDS = {LETTER_WORD_PREFIX + letter: phones for letter, phones in LETTER_PHONES.items()}
SYMBOL_WORDS = {"&": "and", "%": "percent", "@": "at", "+": "plus", "=": "equals", "$": "dollars"}
WORD_ACRONYMS = {
    "ai", "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it",
    "me", "my", "no", "of", "oh", "ok", "on", "or", "so", "to", "up", "us", "we", "hi",
    "ya", "yo", "ah", "eh", "ha", "la", "ma", "pa", "ex", "id", "tv",
}
VOWEL_LETTERS = set("aeiou")
PUNCTUATION = ",;:.!?"
MERIDIEMS = {"am": "ltra ltrm", "pm": "ltrp ltrm"}

TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)*"
    r"|\d+(?:[.,:]\d+)*(?:st|nd|rd|th)?"
    r"|[,;:.!?]"
    r"|[&%@+=$]"
    r"|\S"
)

VOWEL_PHONE = re.compile(r"^(AA|AE|AH|AO|AW|AY|EH|ER|EY|IH|IY|OW|OY|UH|UW)[0-2]$")


def number_words(n):
    if n < 0:
        return "minus " + number_words(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return TENS[tens] + ("" if unit == 0 else " " + ONES[unit])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return ONES[hundreds] + " hundred" + ("" if rest == 0 else " " + number_words(rest))
    for scale, name in SCALES:
        if n >= scale:
            head, rest = divmod(n, scale)
            return number_words(head) + " " + name + ("" if rest == 0 else " " + number_words(rest))
    return str(n)


def year_words(n):
    if 2000 <= n < 2010:
        return "two thousand" + ("" if n == 2000 else " " + ONES[n - 2000])
    high, low = divmod(n, 100)
    if low == 0:
        return number_words(high) + " hundred"
    if low < 10:
        return number_words(high) + " oh " + ONES[low]
    return number_words(high) + " " + number_words(low)


def ordinal_words(n):
    words = number_words(n).split(" ")
    last = words[-1]
    if last in ORDINAL_IRREGULAR:
        last = ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return " ".join(words[:-1] + [last])


def digits_words(digits):
    return " ".join(ONES[int(d)] for d in digits)


def time_words(hours, minutes):
    words = number_words(hours)
    if minutes == 0:
        return words
    if minutes < 10:
        return words + " oh " + ONES[minutes]
    return words + " " + number_words(minutes)


class Rejected(Exception):
    pass


def spell_acronym(token):
    return " ".join(LETTER_WORD_PREFIX + c for c in token.lower())


def is_spelled_acronym(token, whole_text_upper):
    if whole_text_upper or not token.isupper() or len(token) < 2 or len(token) > 6:
        return False
    if token.lower() in WORD_ACRONYMS:
        return False
    if len(token) <= 3:
        return True
    return not (set(token.lower()) & VOWEL_LETTERS)


def expand_number_token(token, strict):
    ordinal = re.fullmatch(r"(\d+)(st|nd|rd|th)", token)
    if ordinal:
        value = int(ordinal.group(1))
        if strict and value > 31:
            raise Rejected(token)
        return ordinal_words(value)
    if re.fullmatch(r"\d{1,3}(,\d{3})+", token):
        if strict:
            raise Rejected(token)
        return number_words(int(token.replace(",", "")))
    decimal = re.fullmatch(r"(\d+)\.(\d+)", token)
    if decimal:
        if strict:
            raise Rejected(token)
        return number_words(int(decimal.group(1))) + " point " + digits_words(decimal.group(2))
    if re.fullmatch(r"\d+", token):
        value = int(token)
        if strict:
            if len(token) > 2:
                raise Rejected(token)
            return number_words(value)
        if len(token) > 4:
            return digits_words(token)
        if len(token) == 4 and 1100 <= value <= 2099:
            return year_words(value)
        if len(token) > 1 and token.startswith("0"):
            return digits_words(token)
        return number_words(value)
    if strict:
        raise Rejected(token)
    return digits_words(re.sub(r"\D", "", token))


def normalize_tokens(text, strict=False):
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text.strip())
    whole_text_upper = text.isupper()
    raw = TOKEN_PATTERN.findall(text)
    output = []
    i = 0
    while i < len(raw):
        token = raw[i]
        following = raw[i + 1] if i + 1 < len(raw) else ""
        if re.fullmatch(r"\d{1,2}:\d{2}", token):
            if strict:
                raise Rejected(token)
            hours, minutes = (int(p) for p in token.split(":"))
            words = time_words(hours, minutes)
            if following.lower() in MERIDIEMS:
                words += " " + MERIDIEMS[following.lower()]
                i += 1
            output.extend(words.split(" "))
        elif token[0].isdigit():
            words = expand_number_token(token, strict)
            output.extend(words.split(" "))
        elif token[0].isalpha():
            if is_spelled_acronym(token, whole_text_upper):
                output.extend(spell_acronym(token).split(" "))
            else:
                output.append(token.lower())
        elif token in SYMBOL_WORDS:
            output.append(SYMBOL_WORDS[token])
        elif token in PUNCTUATION:
            output.append(token)
        elif token in "-/\\\"()[]{}<>*_|~^`#'":
            pass
        elif strict:
            raise Rejected(token)
        i += 1
    return output


def training_words(text):
    try:
        tokens = normalize_tokens(text, strict=True)
    except Rejected:
        return None
    words = [t for t in tokens if t not in PUNCTUATION]
    if not words:
        return None
    if any(not re.fullmatch(r"[a-z]+(?:'[a-z]+)*", w) for w in words):
        return None
    return words


def normalize_text(text):
    return " ".join(normalize_tokens(text, strict=False))


def ensure_nltk_resources():
    import nltk

    for resource, kind in (
        ("averaged_perceptron_tagger_eng", "taggers"),
        ("cmudict", "corpora"),
    ):
        try:
            nltk.data.find("%s/%s" % (kind, resource))
        except LookupError:
            try:
                nltk.data.find("%s/%s.zip" % (kind, resource))
            except LookupError:
                nltk.download(resource, quiet=True)


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


class Lexicon:
    def __init__(self, path=LEXICON_PATH):
        self.entries = {}
        self.variants = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                word, _, phones = line.rstrip("\n").partition("\t")
                if word and phones:
                    phones = phones.split()
                    self.entries.setdefault(word, phones)
                    self.variants.setdefault(word, []).append(phones)
        for word, phones in LETTER_WORDS.items():
            self.entries[word] = list(phones)
            self.variants[word] = [list(phones)]
        self._g2p = None

    @property
    def g2p(self):
        if self._g2p is None:
            ensure_nltk_resources()
            from g2p_en import G2p

            self._g2p = G2p()
        return self._g2p

    def tag(self, words):
        ensure_nltk_resources()
        from nltk import pos_tag

        return [tag for _, tag in pos_tag(words)]

    def predict(self, word):
        g2p = self.g2p
        cmu = g2p.cmu.get(word)
        if cmu:
            return list(cmu[0])
        letters = re.sub(r"[^a-z']", "", word)
        phones = [p for p in g2p.predict(letters) if p != " " and re.fullmatch(r"[A-Z]+[0-2]?", p)]
        if letters.endswith("oo") and phones and phones[-1] in ("OW0", "OW1", "OW2"):
            phones[-1] = "UW1"
        return phones

    def homograph(self, word, tag):
        entry = self.g2p.homograph2features.get(word)
        if entry is None:
            return None
        first, second, first_tag = entry
        chosen = list(first) if tag.startswith(first_tag) else list(second)
        candidates = self.variants.get(word, [])
        if len(candidates) > 1:
            return list(min(candidates, key=lambda c: edit_distance(c, chosen)))
        return chosen

    def word_phones(self, word, tag=None):
        if tag is not None and word in self.g2p.homograph2features:
            return self.homograph(word, tag)
        if word in self.entries:
            return list(self.entries[word])
        return self.predict(word)


def is_vowel_phone(phone):
    return bool(VOWEL_PHONE.match(phone))


def repair_broken_words(tokens, lexicon):
    repaired = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        following = tokens[i + 1] if i + 1 < len(tokens) else None
        if (
            following is not None
            and token not in PUNCTUATION
            and following not in PUNCTUATION
            and token not in lexicon.entries
            and token + following in lexicon.entries
        ):
            repaired.append(token + following)
            i += 2
        else:
            repaired.append(token)
            i += 1
    return repaired


def text_to_phones_and_words(text, lexicon, repair=False):
    tokens = normalize_tokens(text, strict=False)
    if repair:
        tokens = repair_broken_words(tokens, lexicon)
    word_tokens = [t for t in tokens if t not in PUNCTUATION]
    tags = lexicon.tag(word_tokens) if word_tokens else []
    tag_iter = iter(tags)
    pieces = []
    for token in tokens:
        if token in PUNCTUATION:
            pieces.append(None)
        else:
            pieces.append(lexicon.word_phones(token, next(tag_iter)))
    phones = []
    words = []
    for index, piece in enumerate(pieces):
        if piece is None:
            if phones and phones[-1] != "sp":
                phones.append("sp")
            continue
        piece = list(piece)
        if tokens[index] == "the" and index + 1 < len(pieces):
            following = pieces[index + 1]
            if following and is_vowel_phone(following[0]) and piece[-2:] == ["DH", "AH0"]:
                piece[-1] = "IY0"
        start = len(phones)
        phones.extend(piece)
        words.append((tokens[index], start, len(phones)))
    while phones and phones[-1] == "sp":
        phones.pop()
    return phones, words


def text_to_phones(text, lexicon, repair=False):
    return text_to_phones_and_words(text, lexicon, repair)[0]
