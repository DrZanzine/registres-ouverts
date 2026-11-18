# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Normalisation, CER et WER. Les noms (Fictivard, Imaginaire…) sont FICTIFS."""

import unicodedata

import pytest

from registres_ouverts.evaluate.metrics import (
    Score,
    cer,
    char_confusions,
    compare_text,
    corpus_scores,
    rate,
    wer,
)
from registres_ouverts.evaluate.normalize import Normalization, normalize

NFC_E = "é"  # é composé
NFD_E = "é"  # e + accent aigu combinant


# --- Normalisation ---------------------------------------------------------


def test_nfc_always_applied() -> None:
    assert normalize(f"Andr{NFD_E}") == f"Andr{NFC_E}"
    assert Normalization().labels() == ["nfc"]


def test_default_keeps_case_punctuation_spaces_accents() -> None:
    text = "  Né à Fictiville,  le Deux Mai. "
    assert normalize(text) == text


def test_casefold() -> None:
    assert normalize("FICTIVARD Émile", Normalization(casefold=True)) == "fictivard émile"


def test_punctuation() -> None:
    options = Normalization(punctuation=True)
    assert normalize("l'an mil-huit, « fictif ».", options) == "lan milhuit  fictif "


def test_whitespace() -> None:
    options = Normalization(whitespace=True)
    assert normalize(" deux  mai\t1893 ", options) == "deux mai 1893"


def test_diacritics() -> None:
    options = Normalization(diacritics=True)
    assert normalize(f"Fran{chr(0xE7)}ois {NFD_E}t{NFC_E} œuvre", options) == "Francois ete œuvre"


def test_all_options_and_labels() -> None:
    options = Normalization(casefold=True, punctuation=True, whitespace=True, diacritics=True)
    assert options.labels() == ["nfc", "casse", "diacritiques", "ponctuation", "espaces"]
    assert normalize(" Élise-Marie  FICTIVARD. ", options) == "elisemarie fictivard"


# --- Taux ----------------------------------------------------------------------


def test_rate_conventions() -> None:
    assert rate(0, 0) == 0.0
    assert rate(3, 0) is None
    assert rate(1, 4) == 0.25


def test_identical() -> None:
    assert cer("Jean Fictivard", "Jean Fictivard") == 0.0
    assert wer("Jean Fictivard", "Jean Fictivard") == 0.0


def test_simple_cer_wer() -> None:
    # Une substitution sur 9 caractères, un mot faux sur 1.
    assert cer("Fictivard", "Fictivart") == pytest.approx(1 / 9)
    assert wer("né le deux mai", "né la deux mai") == pytest.approx(1 / 4)


def test_cer_can_exceed_one() -> None:
    assert cer("ab", "abcdef") == 2.0


def test_empty_strings() -> None:
    assert cer("", "") == 0.0
    assert wer("", "") == 0.0
    assert cer("", "x") is None
    assert wer("", "x") is None
    assert cer("abc", "") == 1.0
    assert wer("deux mots", "") == 1.0


def test_nfd_vs_nfc_no_error() -> None:
    ref = f"Ren{NFC_E}e Imaginaire"
    hyp = unicodedata.normalize("NFD", ref)
    assert ref != hyp
    comparison = compare_text(ref, hyp)
    assert comparison.cer.errors == 0
    assert comparison.cer.reference_length == len(ref)


def test_normalization_reduces_errors() -> None:
    ref, hyp = "Né à Fictiville.", "ne a fictiville"
    assert cer(ref, hyp) > 0
    options = Normalization(casefold=True, punctuation=True, diacritics=True)
    assert cer(ref, hyp, options) == 0.0


def test_wer_ignores_extra_spaces() -> None:
    assert wer("deux mai", "deux   mai") == 0.0
    assert cer("deux mai", "deux   mai") == pytest.approx(2 / 8)


def test_score_dict_and_sum() -> None:
    comparison = compare_text("abc", "abd")
    data = comparison.cer.to_dict()
    assert data["erreurs"] == 1
    assert data["substitutions"] == 1
    assert data["longueur_reference"] == 3
    total = comparison.cer + Score()
    assert total.rate == pytest.approx(1 / 3)


def test_corpus_micro_average() -> None:
    # Ligne 1 : 1 erreur / 10 ; ligne 2 : 1 erreur / 2. Micro = 2/12, pas (0.1+0.5)/2.
    pairs = [("abcdefghij", "abcdefghiX"), ("ab", "aX")]
    total_cer, total_wer = corpus_scores(pairs)
    assert total_cer.rate == pytest.approx(2 / 12)
    assert total_wer.rate == pytest.approx(2 / 2)


def test_corpus_with_empty_reference_counts_insertions() -> None:
    total_cer, _ = corpus_scores([("abcd", "abcd"), ("", "xy")])
    assert total_cer.errors == 2
    assert total_cer.rate == pytest.approx(2 / 4)


def test_char_confusions() -> None:
    comparisons = [compare_text("nuit", "uuit"), compare_text("non", "uou")]
    counter = char_confusions(comparisons)
    assert counter[("n", "u")] == 3
