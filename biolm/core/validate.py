import re
from typing import List

UNAMBIGUOUS_AA = (
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
)
aa_unambiguous = "ACDEFGHIKLMNPQRSTVWY"
aa_extended = aa_unambiguous + "BXZUO"

dna_unambiguous = "ACTG"
dna_ambiguous = dna_unambiguous + "XNU"


regexes = {
    "empty_or_aa_unambiguous_validator": re.compile(f"^[{aa_unambiguous}]*$"),
    "aa_extended_validator": re.compile(f"^[{aa_extended}]+$"),
    "aa_unambiguous_validator": re.compile(f"^[{aa_unambiguous}]+$"),
    "empty_or_dna_unambiguous_validator": re.compile(f"^[{dna_unambiguous}]*$"),
    "dna_unambiguous_validator": re.compile(f"^[{dna_unambiguous}]+$"),
}


def empty_or_aa_unambiguous_validator(text: str) -> str:
    if not regexes["empty_or_aa_unambiguous_validator"].match(text):
        raise ValueError(
            f"Residues can only be represented with '{aa_unambiguous}' characters"
        )
    return text


def empty_or_dna_unambiguous_validator(text: str) -> str:
    if not regexes["empty_or_dna_unambiguous_validator"].match(text):
        raise ValueError(
            f"Nucleotides can only be represented with '{dna_unambiguous}' characters"
        )
    return text


def aa_extended_validator(text: str) -> str:
    if not regexes["aa_extended_validator"].match(text):
        raise ValueError(
            f"Residues can only be represented with '{aa_extended}' characters"
        )
    return text


def aa_unambiguous_validator(text: str) -> str:
    if not regexes["aa_unambiguous_validator"].match(text):
        raise ValueError(
            f"Residues can only be represented with '{aa_unambiguous}' characters"
        )
    return text


def dna_unambiguous_validator(text: str) -> str:
    if not regexes["dna_unambiguous_validator"].match(text):
        raise ValueError(
            f"Nucleotides can only be represented with '{dna_unambiguous}' characters"
        )
    return text

def pdb_validator(text: str) -> str:
    if "ATOM" not in text:
        raise ValueError("PDB string does not appear to be a valid PDB")
    return text


class PDB:
    def __call__(self, value):
        _ = pdb_validator(value)
class AAUnambiguous:
    def __call__(self, value):
        _ = aa_unambiguous_validator(value)

class AAExtended:
    def __call__(self, value):
        _ = aa_extended_validator(value)

class DNAUnambiguous:
    def __call__(self, value):
        _ = dna_unambiguous_validator(value)

class AAUnambiguousEmpty:
    def __call__(self, value):
        _ = empty_or_aa_unambiguous_validator(value)

class AAUnambiguousPlusExtra:
    def __init__(self, extra: List[str]):
        if not extra:
            raise ValueError("Extra cannot be empty")
        self.extra = extra

    def __call__(self, value: str) -> str:
        text_clean = value
        for ex in self.extra:
            text_clean = text_clean.replace(ex, "")
        aa_unambiguous_validator(text_clean)
        return value


class AAExtendedPlusExtra:
    def __init__(self, extra: List[str]):
        if not extra:
            raise ValueError("Extra cannot be empty")
        self.extra = extra

    def __call__(self, value: str) -> str:
        text_clean = value
        for ex in self.extra:
            text_clean = text_clean.replace(ex, "")
        aa_extended_validator(text_clean)
        return value


class SingleOccurrenceOf:
    def __init__(self, single_token: str):
        self.single_token = single_token

    def __call__(self, value: str) -> str:
        count = value.count(self.single_token)
        if count != 1:
            raise ValueError(
                f"Expected a single occurrence of '{self.single_token}', got {count}"
            )
        return value


class SingleOrMoreOccurrencesOf:
    def __init__(self, token: str):
        self.token = token

    def __call__(self, value: str) -> str:
        count = value.count(self.token)
        if count < 1:
            raise ValueError(
                f"Expected at least one occurrence of '{self.token}', got none"
            )
        return value
