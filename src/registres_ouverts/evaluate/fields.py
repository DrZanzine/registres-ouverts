# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Évaluation par champ des actes extraits (JSON de configs/acte.schema.json).

Un *champ* est tout objet ``{"brut", "normalise", "confiance"}`` de l'acte
(``enfant.nom``, ``pere.age``, ``temoins[1].profession``…). La valeur comparée
est ``normalise``, à défaut ``brut`` ; elle est convertie en texte puis
normalisée (NFC + options choisies). Une valeur ``null`` ou vide est
« absente ».

Pour chaque champ, en comparant référence (R) et hypothèse (H) :

- ``correct`` : R et H présentes et égales → vrai positif ;
- ``faux`` : R et H présentes et différentes → faux positif ET faux négatif ;
- ``manquant`` : R présente, H absente → faux négatif ;
- ``en_trop`` : R absente, H présente → faux positif ;
- ``absent`` : les deux absentes → non compté.

Précision = VP / (VP + FP), rappel = VP / (VP + FN), F1 = moyenne harmonique ;
``None`` quand le dénominateur est nul. Les témoins sont regroupés sous un même
nom de champ (``temoins.nom``), appariés par rang.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from registres_ouverts.evaluate.normalize import Normalization

LEAF_KEYS = {"brut", "normalise", "confiance"}
_INDEX = re.compile(r"\[\d+\]")


class ActError(ValueError):
    """Fichier d'acte illisible ou invalide."""


class Outcome(StrEnum):
    """Issue de la comparaison d'un champ."""

    CORRECT = "correct"
    WRONG = "faux"
    MISSING = "manquant"
    EXTRA = "en_trop"
    ABSENT = "absent"


@dataclass(frozen=True, slots=True)
class Leaf:
    """Valeur d'un champ dans un acte."""

    value: str | None
    confidence: float | None


def iter_leaves(node: Any, path: str = "") -> Iterator[tuple[str, dict]]:
    """Parcourt les champs ``{brut, normalise, confiance}`` d'un acte."""
    if isinstance(node, dict):
        if LEAF_KEYS <= set(node):
            yield path, node
            return
        for key, value in node.items():
            yield from iter_leaves(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from iter_leaves(value, f"{path}[{index}]")


def leaf_value(leaf: dict, normalization: Normalization) -> str | None:
    """Valeur comparée : ``normalise``, à défaut ``brut`` ; ``None`` si absente."""
    raw = leaf.get("normalise")
    if raw is None:
        raw = leaf.get("brut")
    if raw is None:
        return None
    text = normalization(str(raw)).strip()
    return text or None


def extract_fields(act: dict, normalization: Normalization | None = None) -> dict[str, Leaf]:
    """Champs d'un acte, indexés par chemin."""
    normalization = normalization or Normalization()
    fields: dict[str, Leaf] = {}
    for path, leaf in iter_leaves(act):
        confidence = leaf.get("confiance")
        fields[path] = Leaf(
            leaf_value(leaf, normalization),
            float(confidence) if isinstance(confidence, int | float) else None,
        )
    return fields


def field_group(path: str) -> str:
    """Nom de champ agrégé : ``temoins[1].nom`` → ``temoins.nom``."""
    return _INDEX.sub("", path)


def classify(reference: str | None, hypothesis: str | None) -> Outcome:
    """Issue de la comparaison de deux valeurs (déjà normalisées)."""
    if reference is None:
        return Outcome.ABSENT if hypothesis is None else Outcome.EXTRA
    if hypothesis is None:
        return Outcome.MISSING
    return Outcome.CORRECT if reference == hypothesis else Outcome.WRONG


@dataclass(frozen=True, slots=True)
class FieldComparison:
    """Comparaison d'un champ d'un acte."""

    act: str
    path: str
    reference: str | None
    hypothesis: str | None
    confidence: float | None
    outcome: Outcome

    def to_dict(self) -> dict:
        """Représentation JSON."""
        return {
            "acte": self.act,
            "champ": self.path,
            "reference": self.reference,
            "hypothese": self.hypothesis,
            "confiance": self.confidence,
            "issue": str(self.outcome),
        }


def compare_acts(
    reference: dict,
    hypothesis: dict | None,
    normalization: Normalization | None = None,
    act: str = "",
) -> list[FieldComparison]:
    """Compare champ à champ deux actes ; ``hypothesis=None`` : acte non produit."""
    ref_fields = extract_fields(reference, normalization)
    hyp_fields = extract_fields(hypothesis, normalization) if hypothesis is not None else {}
    empty = Leaf(None, None)
    paths = list(ref_fields) + [p for p in hyp_fields if p not in ref_fields]
    comparisons = []
    for path in paths:
        ref, hyp = ref_fields.get(path, empty), hyp_fields.get(path, empty)
        comparisons.append(
            FieldComparison(
                act, path, ref.value, hyp.value, hyp.confidence, classify(ref.value, hyp.value)
            )
        )
    return comparisons


@dataclass(slots=True)
class FieldStats:
    """Vrais positifs, faux positifs et faux négatifs d'un champ."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    support: int = 0  # valeurs présentes dans la référence

    def add(self, outcome: Outcome) -> None:
        """Comptabilise une issue."""
        if outcome is Outcome.CORRECT:
            self.true_positives += 1
        if outcome in (Outcome.WRONG, Outcome.EXTRA):
            self.false_positives += 1
        if outcome in (Outcome.WRONG, Outcome.MISSING):
            self.false_negatives += 1
        if outcome in (Outcome.CORRECT, Outcome.WRONG, Outcome.MISSING):
            self.support += 1

    @property
    def precision(self) -> float | None:
        """VP / (VP + FP)."""
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total else None

    @property
    def recall(self) -> float | None:
        """VP / (VP + FN)."""
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total else None

    @property
    def f1(self) -> float | None:
        """Moyenne harmonique de la précision et du rappel."""
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if p + r else 0.0

    def to_dict(self) -> dict:
        """Représentation JSON."""
        return {
            "precision": self.precision,
            "rappel": self.recall,
            "f1": self.f1,
            "vrais_positifs": self.true_positives,
            "faux_positifs": self.false_positives,
            "faux_negatifs": self.false_negatives,
            "support": self.support,
        }


@dataclass(slots=True)
class FieldsResult:
    """Résultat de l'évaluation par champ sur un ensemble d'actes."""

    normalization: Normalization
    comparisons: list[FieldComparison] = field(default_factory=list)
    missing_acts: list[str] = field(default_factory=list)
    extra_acts: list[str] = field(default_factory=list)

    def by_field(self) -> dict[str, FieldStats]:
        """Statistiques par champ agrégé, dans l'ordre d'apparition."""
        stats: dict[str, FieldStats] = {}
        for comparison in self.comparisons:
            stats.setdefault(field_group(comparison.path), FieldStats()).add(comparison.outcome)
        return stats

    def overall(self) -> FieldStats:
        """Statistiques globales (micro) sur tous les champs."""
        total = FieldStats()
        for comparison in self.comparisons:
            total.add(comparison.outcome)
        return total

    def discrepancies(self) -> list[FieldComparison]:
        """Écarts : champs ni corrects ni absents des deux côtés."""
        return [c for c in self.comparisons if c.outcome not in (Outcome.CORRECT, Outcome.ABSENT)]

    def confidence_pairs(self) -> list[tuple[float, bool]]:
        """Couples (confiance, correct) des valeurs produites avec une confiance."""
        return [
            (c.confidence, c.outcome is Outcome.CORRECT)
            for c in self.comparisons
            if c.hypothesis is not None and c.confidence is not None
        ]

    def to_dict(self) -> dict:
        """Représentation JSON."""
        return {
            "normalisation": self.normalization.labels(),
            "actes": len({c.act for c in self.comparisons}),
            "actes_manquants": self.missing_acts,
            "actes_en_trop": self.extra_acts,
            "global": self.overall().to_dict(),
            "champs": {name: stats.to_dict() for name, stats in self.by_field().items()},
            "ecarts": [c.to_dict() for c in self.discrepancies()],
        }


def load_acts(path: Path) -> dict[str, dict]:
    """Charge un acte (fichier JSON) ou tous les ``*.json`` d'un dossier.

    Les actes sont indexés par leur ``id`` si tous en ont un (unique), sinon
    par nom de fichier sans extension.
    """
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    acts: list[tuple[str, dict]] = []
    for file in files:
        try:
            act = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ActError(f"{file} : {error}") from error
        if not isinstance(act, dict):
            raise ActError(f"{file} : un objet JSON (un acte) est attendu")
        acts.append((file.stem, act))
    ids = [act.get("id") for _, act in acts]
    if all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids):
        return {act["id"]: act for _, act in acts}
    return dict(acts)


def evaluate_acts(
    reference: dict[str, dict],
    hypothesis: dict[str, dict],
    normalization: Normalization | None = None,
) -> FieldsResult:
    """Évalue des actes appariés par clé (``id`` ou nom de fichier).

    Un acte de référence sans hypothèse est signalé et tous ses champs
    renseignés comptent comme manquants ; un acte en trop est signalé mais non
    compté.
    """
    normalization = normalization or Normalization()
    result = FieldsResult(normalization=normalization)
    for key, ref in reference.items():
        hyp = hypothesis.get(key)
        if hyp is None:
            result.missing_acts.append(key)
        result.comparisons += compare_acts(ref, hyp, normalization, act=key)
    result.extra_acts = [key for key in hypothesis if key not in reference]
    return result


def evaluate_paths(
    reference: Path, hypothesis: Path, normalization: Normalization | None = None
) -> FieldsResult:
    """Évalue deux fichiers d'actes, ou deux dossiers d'actes."""
    if reference.is_dir() != hypothesis.is_dir():
        raise ActError("référence et hypothèse : deux fichiers ou deux dossiers")
    if not reference.is_dir():
        ref, hyp = load_acts(reference), load_acts(hypothesis)
        # Deux fichiers isolés : on les compare quels que soient leurs identifiants.
        (ref_key, ref_act), (_, hyp_act) = next(iter(ref.items())), next(iter(hyp.items()))
        return evaluate_acts({ref_key: ref_act}, {ref_key: hyp_act}, normalization)
    return evaluate_acts(load_acts(reference), load_acts(hypothesis), normalization)
