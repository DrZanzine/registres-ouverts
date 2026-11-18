# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Courbe de seuil (données fictives)."""

import json
from pathlib import Path

import pytest

from registres_ouverts.evaluate.threshold import (
    ThresholdError,
    ThresholdPoint,
    best_threshold,
    grid,
    load_pairs,
    threshold_curve,
)

PAIRS = [(0.95, True), (0.9, True), (0.8, False), (0.7, True), (0.5, False)]


def test_curve_on_observed_confidences() -> None:
    points = threshold_curve(PAIRS)
    assert [p.threshold for p in points] == [0.5, 0.7, 0.8, 0.9, 0.95]
    first, middle, last = points[0], points[2], points[-1]
    assert (first.accepted, first.correct, first.coverage) == (5, 3, 1.0)
    assert first.precision == pytest.approx(0.6)
    assert (middle.accepted, middle.correct) == (3, 2)
    assert middle.precision == pytest.approx(2 / 3)
    assert last.coverage == pytest.approx(0.2)
    assert last.precision == 1.0


def test_curve_explicit_thresholds_and_empty_acceptance() -> None:
    points = threshold_curve(PAIRS, thresholds=[1.0, 0.0])
    assert [p.threshold for p in points] == [0.0, 1.0]
    assert points[1].accepted == 0
    assert points[1].precision is None
    assert points[1].coverage == 0.0


def test_curve_empty_input() -> None:
    assert threshold_curve([]) == []
    point = threshold_curve([], thresholds=[0.5])[0]
    assert point.coverage is None and point.precision is None


def test_curve_rejects_invalid_confidence() -> None:
    for bad in [1.5, -0.1, float("nan")]:
        with pytest.raises(ThresholdError):
            threshold_curve([(bad, True)])


def test_best_threshold() -> None:
    points = threshold_curve(PAIRS)
    assert best_threshold(points, 0.99).threshold == 0.9
    assert best_threshold(points, 0.5).threshold == 0.5
    assert best_threshold(threshold_curve([(0.5, False)]), 0.99) is None


def test_point_dict() -> None:
    data = ThresholdPoint(0.5, 2, 1, 4).to_dict()
    assert data == {
        "seuil": 0.5,
        "acceptees": 2,
        "correctes": 1,
        "total": 4,
        "couverture": 0.5,
        "precision": 0.5,
    }


def test_grid() -> None:
    assert grid(0.25) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert grid(0.1)[-1] == 1.0 and len(grid(0.1)) == 11
    assert grid(0.3) == [0.0, 0.3, 0.6, 0.9, 1.0]
    for bad in [0, -1, 2]:
        with pytest.raises(ThresholdError):
            grid(bad)


def test_load_csv(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    path.write_text('confiance,correct\n0.9,1\n"0,4",faux\n0.7,true\n', "utf-8")
    assert load_pairs(path) == [(0.9, True), (0.4, False), (0.7, True)]


def test_load_json(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps([{"confiance": 0.9, "correct": True}, [0.2, 0]]), "utf-8")
    assert load_pairs(path) == [(0.9, True), (0.2, False)]


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("a.csv", "score,ok\n0.5,1\n"),
        ("b.csv", "confiance,correct\n0.5,peut-etre\n"),
        ("c.csv", "confiance,correct\nabc,1\n"),
        ("d.csv", "confiance,correct\n2,1\n"),
        ("e.json", "{}"),
        ("f.json", "[[0.5]]"),
        ("g.json", "[oups"),
    ],
)
def test_load_errors(tmp_path: Path, name: str, content: str) -> None:
    path = tmp_path / name
    path.write_text(content, "utf-8")
    with pytest.raises(ThresholdError):
        load_pairs(path)


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ThresholdError):
        load_pairs(tmp_path / "absent.csv")
