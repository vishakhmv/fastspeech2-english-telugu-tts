from . import telugu

PAD = "_"
SILENCE = "sp"
SPOKEN_NOISE = "spn"

ARPABET_CONSONANTS = [
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG",
    "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
]
ARPABET_VOWELS = [
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
]
ARPABET_PHONES = ARPABET_CONSONANTS + [v + str(stress) for v in ARPABET_VOWELS for stress in range(3)]

LANGUAGE_PHONES = {
    "english": ARPABET_PHONES + [SILENCE, SPOKEN_NOISE],
    "telugu": telugu.PHONES + [SILENCE],
}


def get_symbols(language):
    return [PAD] + LANGUAGE_PHONES[language]


def symbol_to_id(language):
    return {symbol: index for index, symbol in enumerate(get_symbols(language))}


def phones_to_sequence(phones, language):
    table = symbol_to_id(language)
    unknown = [p for p in phones if p not in table]
    if unknown:
        raise KeyError("unknown phones for %s: %s" % (language, sorted(set(unknown))))
    return [table[p] for p in phones]


def sequence_to_phones(sequence, language):
    symbols = get_symbols(language)
    return [symbols[i] for i in sequence if i != 0]


def parse_phone_field(field):
    return field.strip().lstrip("{").rstrip("}").split()


def format_phone_field(phones):
    return "{" + " ".join(phones) + "}"
