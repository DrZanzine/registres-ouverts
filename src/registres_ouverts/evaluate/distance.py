# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Distance d'édition (Levenshtein) et alignement, en Python pur.

Les séquences comparées sont des chaînes (alignement caractère par caractère,
pour le CER) ou des listes de mots (pour le WER). Chaque opération coûte 1 :
substitution, insertion, suppression.

Conventions de sens : la *référence* est la vérité, l'*hypothèse* la sortie du
moteur. Une *insertion* est un élément présent dans l'hypothèse mais absent de
la référence ; une *suppression* un élément de la référence absent de
l'hypothèse.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum


class OpKind(StrEnum):
    """Type d'opération d'alignement."""

    EQUAL = "equal"
    SUBSTITUTE = "substitute"
    INSERT = "insert"
    DELETE = "delete"


@dataclass(frozen=True, slots=True)
class EditOp:
    """Une opération de l'alignement référence → hypothèse.

    ``ref`` vaut ``None`` pour une insertion, ``hyp`` vaut ``None`` pour une
    suppression. Les indices pointent dans les séquences d'origine.
    """

    kind: OpKind
    ref: str | None
    hyp: str | None
    ref_index: int | None
    hyp_index: int | None


def levenshtein(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Distance de Levenshtein entre deux séquences (mémoire linéaire).

    >>> levenshtein("chat", "chats")
    1
    >>> levenshtein(["un", "acte"], ["une", "acte"])
    1
    """
    if len(ref) < len(hyp):
        # Une ligne de la longueur de la séquence la plus courte suffit.
        ref, hyp = hyp, ref
    if not hyp:
        return len(ref)
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(
                min(
                    previous[j] + 1,  # suppression
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (r != h),  # substitution ou égalité
                )
            )
        previous = current
    return previous[-1]


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[EditOp]:
    """Alignement optimal de ``ref`` sur ``hyp``.

    Renvoie la liste ordonnée des opérations ; le nombre d'opérations autres
    que ``EQUAL`` est égal à la distance de Levenshtein. En cas d'égalité de
    coût, la diagonale (égalité ou substitution) est préférée, puis la
    suppression, puis l'insertion : le résultat est déterministe.
    """
    n, m = len(ref), len(hyp)
    # Matrice complète des coûts : les lignes d'un registre sont courtes.
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        row, above = cost[i], cost[i - 1]
        r = ref[i - 1]
        for j in range(1, m + 1):
            row[j] = min(above[j] + 1, row[j - 1] + 1, above[j - 1] + (r != hyp[j - 1]))

    ops: list[EditOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            kind = OpKind.EQUAL if ref[i - 1] == hyp[j - 1] else OpKind.SUBSTITUTE
            ops.append(EditOp(kind, ref[i - 1], hyp[j - 1], i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(EditOp(OpKind.DELETE, ref[i - 1], None, i - 1, None))
            i -= 1
        else:
            ops.append(EditOp(OpKind.INSERT, None, hyp[j - 1], None, j - 1))
            j -= 1
    ops.reverse()
    return ops


@dataclass(frozen=True, slots=True)
class OpCounts:
    """Décompte des opérations d'un alignement."""

    equal: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0

    @property
    def distance(self) -> int:
        """Distance d'édition correspondante."""
        return self.substitutions + self.insertions + self.deletions

    def __add__(self, other: OpCounts) -> OpCounts:
        return OpCounts(
            self.equal + other.equal,
            self.substitutions + other.substitutions,
            self.insertions + other.insertions,
            self.deletions + other.deletions,
        )


def count_ops(ops: Iterable[EditOp]) -> OpCounts:
    """Compte les opérations par type."""
    counter = Counter(op.kind for op in ops)
    return OpCounts(
        equal=counter[OpKind.EQUAL],
        substitutions=counter[OpKind.SUBSTITUTE],
        insertions=counter[OpKind.INSERT],
        deletions=counter[OpKind.DELETE],
    )


def confusions(ops: Iterable[EditOp]) -> Counter[tuple[str, str]]:
    """Compte les erreurs sous forme de couples (référence, hypothèse).

    Une insertion est notée ``("", hyp)``, une suppression ``(ref, "")``.
    Utile pour repérer les confusions récurrentes du moteur (``u``/``n``…).
    """
    counter: Counter[tuple[str, str]] = Counter()
    for op in ops:
        if op.kind is not OpKind.EQUAL:
            counter[(op.ref or "", op.hyp or "")] += 1
    return counter
