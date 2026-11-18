# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Courbe de seuil : part de valeurs acceptées et précision selon la confiance.

À partir de couples (confiance, correct), pour chaque seuil *s* :

- valeurs acceptées = celles dont la confiance est ≥ *s* (validées
  automatiquement) ; les autres partent en relecture humaine ;
- couverture = acceptées / total ;
- précision = correctes parmi les acceptées / acceptées (``None`` si aucune).

Le seuil d'envoi en relecture est le plus petit seuil dont la précision atteint
la cible (par exemple 0,99) : c'est celui qui accepte le plus de valeurs.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


class ThresholdError(ValueError):
    """Données de confiance invalides."""


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    """Un point de la courbe."""

    threshold: float
    accepted: int
    correct: int
    total: int

    @property
    def coverage(self) -> float | None:
        """Part des valeurs acceptées."""
        return self.accepted / self.total if self.total else None

    @property
    def precision(self) -> float | None:
        """Précision parmi les valeurs acceptées."""
        return self.correct / self.accepted if self.accepted else None

    def to_dict(self) -> dict:
        """Représentation JSON."""
        return {
            "seuil": self.threshold,
            "acceptees": self.accepted,
            "correctes": self.correct,
            "total": self.total,
            "couverture": self.coverage,
            "precision": self.precision,
        }


def _check(pairs: Iterable[tuple[float, bool]]) -> list[tuple[float, bool]]:
    checked = []
    for confidence, correct in pairs:
        if not isinstance(confidence, int | float) or math.isnan(confidence):
            raise ThresholdError(f"confiance invalide : {confidence!r}")
        if not 0 <= confidence <= 1:
            raise ThresholdError(f"confiance hors de [0, 1] : {confidence!r}")
        checked.append((float(confidence), bool(correct)))
    return checked


def threshold_curve(
    pairs: Iterable[tuple[float, bool]], thresholds: Sequence[float] | None = None
) -> list[ThresholdPoint]:
    """Courbe de seuil, par seuils croissants.

    Sans ``thresholds``, les seuils sont les valeurs de confiance distinctes
    observées : la courbe est alors exacte (aucun point intermédiaire ne
    change le résultat).
    """
    data = sorted(_check(pairs), reverse=True)
    if thresholds is None:
        thresholds = sorted({confidence for confidence, _ in data})
    total = len(data)
    points = []
    for threshold in sorted(thresholds):
        accepted = [correct for confidence, correct in data if confidence >= threshold]
        points.append(ThresholdPoint(threshold, len(accepted), sum(accepted), total))
    return points


def grid(step: float) -> list[float]:
    """Seuils régulièrement espacés de 0 à 1 inclus."""
    if not 0 < step <= 1:
        raise ThresholdError(f"pas invalide : {step!r} (attendu dans ]0, 1])")
    count = math.floor(1 / step + 1e-9)
    values = [round(i * step, 10) for i in range(count + 1)]
    if values[-1] < 1:
        values.append(1.0)
    return values


def best_threshold(points: Iterable[ThresholdPoint], target: float) -> ThresholdPoint | None:
    """Plus petit seuil dont la précision atteint ``target`` ; ``None`` sinon."""
    for point in sorted(points, key=lambda p: p.threshold):
        if point.precision is not None and point.precision >= target:
            return point
    return None


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "vrai", "oui", "correct"}:
        return True
    if text in {"0", "false", "faux", "non", "incorrect"}:
        return False
    raise ThresholdError(f"booléen attendu pour « correct » : {value!r}")


def _parse_float(value: object) -> float:
    try:
        return float(str(value).replace(",", "."))
    except ValueError as error:
        raise ThresholdError(f"nombre attendu pour « confiance » : {value!r}") from error


def load_pairs(path: Path) -> list[tuple[float, bool]]:
    """Charge des couples (confiance, correct) depuis un CSV ou un JSON.

    - CSV avec en-tête ``confiance,correct`` (``correct`` : 1/0, vrai/faux,
      true/false, oui/non) ;
    - JSON : liste de ``{"confiance": …, "correct": …}`` ou de paires
      ``[confiance, correct]``.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ThresholdError(f"{path} : {error}") from error
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(content)
        except json.JSONDecodeError as error:
            raise ThresholdError(f"{path} : {error}") from error
        if not isinstance(data, list):
            raise ThresholdError(f"{path} : liste JSON attendue")
        pairs = []
        for item in data:
            if isinstance(item, dict) and {"confiance", "correct"} <= set(item):
                pairs.append((_parse_float(item["confiance"]), _parse_bool(item["correct"])))
            elif isinstance(item, list) and len(item) == 2:
                pairs.append((_parse_float(item[0]), _parse_bool(item[1])))
            else:
                raise ThresholdError(f"{path} : élément invalide {item!r}")
        return _check(pairs)
    reader = csv.DictReader(content.splitlines())
    if not reader.fieldnames or not {"confiance", "correct"} <= set(reader.fieldnames):
        raise ThresholdError(f"{path} : colonnes « confiance » et « correct » attendues")
    return _check((_parse_float(row["confiance"]), _parse_bool(row["correct"])) for row in reader)
