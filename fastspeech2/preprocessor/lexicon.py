import re
from collections import defaultdict

FLOAT = re.compile(r"^-?\d+(\.\d+)?$")


def read_mfa_dictionary(path):
    entries = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) == 1:
                fields = line.split()
            word = fields[0]
            rest = fields[1:]
            probability = 1.0
            index = 0
            numbers = []
            while index < len(rest) and FLOAT.match(rest[index]):
                numbers.append(float(rest[index]))
                index += 1
            if numbers:
                probability = numbers[0]
            phones = " ".join(rest[index:]).split()
            if phones:
                entries[word].append((probability, phones))
    return entries


def best_pronunciations(entries):
    best = {}
    for word, variants in entries.items():
        best[word] = max(variants, key=lambda item: item[0])[1]
    return best


def write_runtime_lexicon(entries, path, variant_words=()):
    variant_words = set(variant_words)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for word in sorted(entries):
            variants = sorted(entries[word], key=lambda item: -item[0])
            if word not in variant_words:
                variants = variants[:1]
            seen = []
            for _, phones in variants:
                if phones not in seen:
                    seen.append(phones)
                    f.write("%s\t%s\n" % (word, " ".join(phones)))


def write_alignment_dictionary(original_path, extra_pronunciations, out_path):
    with open(original_path, encoding="utf-8") as f:
        original = f.read()
    if not original.endswith("\n"):
        original += "\n"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(original)
        for word in sorted(extra_pronunciations):
            f.write("%s\t0.99\t0.1\t1.0\t1.0\t%s\n" % (word, " ".join(extra_pronunciations[word])))


def write_simple_dictionary(pronunciations, out_path):
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for word in sorted(pronunciations):
            f.write("%s\t%s\n" % (word, " ".join(pronunciations[word])))
