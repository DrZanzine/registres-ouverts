# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Distance de Levenshtein et alignement (textes fictifs)."""

import itertools
import random

import pytest

from registres_ouverts.evaluate.distance import (
    EditOp,
    OpCounts,
    OpKind,
    align,
    confusions,
    count_ops,
    levenshtein,
)


@pytest.mark.parametrize(
    ("ref", "hyp", "expected"),
    [
        ("", "", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("abc", "abc", 0),
        ("chat", "chats", 1),
        ("kitten", "sitting", 3),
        ("Fictivard", "Fictivart", 1),
        ("flaw", "lawn", 2),
        ("é", "e", 1),
    ],
)
def test_levenshtein(ref: str, hyp: str, expected: int) -> None:
    assert levenshtein(ref, hyp) == expected
    assert levenshtein(hyp, ref) == expected


def test_levenshtein_on_words() -> None:
    assert levenshtein(["né", "le", "deux"], ["né", "la", "deux", "mai"]) == 2


def _apply(ops: list[EditOp]) -> tuple[str, str]:
    """Reconstruit référence et hypothèse depuis l'alignement."""
    ref = "".join(op.ref for op in ops if op.ref is not None)
    hyp = "".join(op.hyp for op in ops if op.hyp is not None)
    return ref, hyp


def test_align_ops_kinds() -> None:
    ops = align("abc", "axcd")
    assert [op.kind for op in ops] == [
        OpKind.EQUAL,
        OpKind.SUBSTITUTE,
        OpKind.EQUAL,
        OpKind.INSERT,
    ]
    assert ops[1] == EditOp(OpKind.SUBSTITUTE, "b", "x", 1, 1)
    assert ops[3] == EditOp(OpKind.INSERT, None, "d", None, 3)


def test_align_deletion() -> None:
    ops = align("abc", "ac")
    assert count_ops(ops) == OpCounts(equal=2, substitutions=0, insertions=0, deletions=1)
    deleted = [op for op in ops if op.kind is OpKind.DELETE]
    assert deleted == [EditOp(OpKind.DELETE, "b", None, 1, None)]


def test_align_empty() -> None:
    assert align("", "") == []
    assert [op.kind for op in align("", "ab")] == [OpKind.INSERT, OpKind.INSERT]
    assert [op.kind for op in align("ab", "")] == [OpKind.DELETE, OpKind.DELETE]


def test_align_consistent_with_distance_random() -> None:
    rng = random.Random(1893)
    alphabet = "abé "
    for _ in range(300):
        ref = "".join(rng.choices(alphabet, k=rng.randint(0, 8)))
        hyp = "".join(rng.choices(alphabet, k=rng.randint(0, 8)))
        ops = align(ref, hyp)
        assert count_ops(ops).distance == levenshtein(ref, hyp)
        assert _apply(ops) == (ref, hyp)


def test_levenshtein_matches_bruteforce_small() -> None:
    """Contrôle croisé avec une définition récursive naïve."""

    def naive(a: str, b: str) -> int:
        if not a or not b:
            return len(a) + len(b)
        return min(
            naive(a[1:], b) + 1,
            naive(a, b[1:]) + 1,
            naive(a[1:], b[1:]) + (a[0] != b[0]),
        )

    for a, b in itertools.product(["", "a", "ab", "ba", "aab"], repeat=2):
        assert levenshtein(a, b) == naive(a, b)


def test_counts_addition() -> None:
    total = OpCounts(1, 2, 3, 4) + OpCounts(1, 1, 1, 1)
    assert total == OpCounts(2, 3, 4, 5)
    assert total.distance == 12


def test_confusions() -> None:
    ops = align("nuit", "uuit") + align("ab", "a") + align("a", "ab")
    counter = confusions(ops)
    assert counter[("n", "u")] == 1
    assert counter[("b", "")] == 1
    assert counter[("", "b")] == 1
