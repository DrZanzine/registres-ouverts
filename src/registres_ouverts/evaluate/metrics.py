# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Taux d'erreur par caractère (CER) et par mot (WER).

CER = (substitutions + insertions + suppressions) / nombre de caractères de la
référence, après normalisation. Le WER est le même calcul sur les mots
(séparés par les espaces). Le taux peut dépasser 1 quand l'hypothèse contient
beaucoup d'insertions.

Agrégation globale « micro » : somme des distances / somme des longueurs de
référence, et non moyenne des taux par ligne (une ligne courte pèserait autant
qu'une ligne longue).

Référence vide : le taux vaut 0 si l'hypothèse est vide aussi, il est indéfini
(``None``) sinon ; les insertions sont tout de même comptées dans le global.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from registres_ouverts.evaluate.distance import EditOp, OpCounts, align, confusions, count_ops
from registres_ouverts.evaluate.normalize import Normalization


def rate(errors: int, reference_length: int) -> float | None:
    """Taux d'erreur ; ``None`` si la référence est vide et qu'il y a des erreurs."""
    if reference_length == 0:
        return 0.0 if errors == 0 else None
    return errors / reference_length


def words(text: str) -> list[str]:
    """Découpe en mots sur les espaces (toute suite d'espaces sépare)."""
    return text.split()


@dataclass(slots=True)
class Score:
    """Décompte d'erreurs sur une unité (caractères ou mots)."""

    counts: OpCounts = field(default_factory=OpCounts)
    reference_length: int = 0
    hypothesis_length: int = 0

    @property
    def errors(self) -> int:
        """Distance d'édition."""
        return self.counts.distance

    @property
    def rate(self) -> float | None:
        """Taux d'erreur (CER ou WER)."""
        return rate(self.errors, self.reference_length)

    def __add__(self, other: Score) -> Score:
        return Score(
            self.counts + other.counts,
            self.reference_length + other.reference_length,
            self.hypothesis_length + other.hypothesis_length,
        )

    def to_dict(self) -> dict:
        """Représentation JSON."""
        return {
            "taux": self.rate,
            "erreurs": self.errors,
            "substitutions": self.counts.substitutions,
            "insertions": self.counts.insertions,
            "suppressions": self.counts.deletions,
            "longueur_reference": self.reference_length,
            "longueur_hypothese": self.hypothesis_length,
        }


@dataclass(slots=True)
class TextComparison:
    """Comparaison d'une référence et d'une hypothèse (une ligne, en général)."""

    reference: str
    hypothesis: str
    char_ops: list[EditOp]
    cer: Score
    wer: Score


def compare_text(
    reference: str, hypothesis: str, normalization: Normalization | None = None
) -> TextComparison:
    """Compare deux textes après normalisation : CER, WER et alignement."""
    normalization = normalization or Normalization()
    ref = normalization(reference)
    hyp = normalization(hypothesis)
    char_ops = align(ref, hyp)
    ref_words, hyp_words = words(ref), words(hyp)
    word_ops = align(ref_words, hyp_words)
    return TextComparison(
        reference=ref,
        hypothesis=hyp,
        char_ops=char_ops,
        cer=Score(count_ops(char_ops), len(ref), len(hyp)),
        wer=Score(count_ops(word_ops), len(ref_words), len(hyp_words)),
    )


def cer(
    reference: str, hypothesis: str, normalization: Normalization | None = None
) -> float | None:
    """CER d'un couple de textes."""
    return compare_text(reference, hypothesis, normalization).cer.rate


def wer(
    reference: str, hypothesis: str, normalization: Normalization | None = None
) -> float | None:
    """WER d'un couple de textes."""
    return compare_text(reference, hypothesis, normalization).wer.rate


def corpus_scores(
    pairs: Iterable[tuple[str, str]], normalization: Normalization | None = None
) -> tuple[Score, Score]:
    """CER et WER globaux (micro) sur des couples (référence, hypothèse)."""
    total_cer, total_wer = Score(), Score()
    for reference, hypothesis in pairs:
        comparison = compare_text(reference, hypothesis, normalization)
        total_cer += comparison.cer
        total_wer += comparison.wer
    return total_cer, total_wer


def char_confusions(comparisons: Iterable[TextComparison]) -> Counter[tuple[str, str]]:
    """Confusions de caractères cumulées sur plusieurs comparaisons."""
    counter: Counter[tuple[str, str]] = Counter()
    for comparison in comparisons:
        counter.update(confusions(comparison.char_ops))
    return counter
