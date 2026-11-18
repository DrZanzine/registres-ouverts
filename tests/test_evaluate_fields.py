# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Évaluation par champ des actes.

Toutes les valeurs sont FICTIVES (Fictivard, Imaginaire, Fictiville).
"""

import copy
import json
from pathlib import Path

import pytest

from registres_ouverts.evaluate.fields import (
    ActError,
    FieldStats,
    Outcome,
    classify,
    compare_acts,
    evaluate_acts,
    evaluate_paths,
    extract_fields,
    field_group,
    iter_leaves,
    load_acts,
)
from registres_ouverts.evaluate.normalize import Normalization

ROOT = Path(__file__).resolve().parents[1]
MODELE = json.loads((ROOT / "data" / "reference" / "MODELE.json").read_text(encoding="utf-8"))


def _set(field: dict, brut, normalise=None, confiance=None) -> None:
    field.update(brut=brut, normalise=normalise, confiance=confiance)


@pytest.fixture
def reference() -> dict:
    """Acte de référence fictif."""
    act = copy.deepcopy(MODELE)
    act["id"] = "fictif-N-1893-001"
    _set(act["numero_acte"], "un", 1)
    _set(act["date_acte"], "trois mai mil huit cent quatre-vingt-treize", "1893-05-03")
    _set(act["enfant"]["nom"], "Fictivard", "Fictivard")
    _set(act["enfant"]["prenoms"], "Jean Bte", "Jean-Baptiste")
    _set(act["pere"]["nom"], "Fictivard", "Fictivard")
    _set(act["pere"]["age"], "trente", 30)
    _set(act["mere"]["nom"], "Imaginaire", "Imaginaire")
    _set(act["temoins"][0]["nom"], "Inventé", "Inventé")
    return act


@pytest.fixture
def hypothesis(reference: dict) -> dict:
    """Sortie d'extraction fictive avec des erreurs connues."""
    act = copy.deepcopy(reference)
    for _, leaf in _leaves(act):
        leaf["confiance"] = 0.9
    _set(act["enfant"]["prenoms"], "Jean Bte", "Jean-Baptiste", 0.95)  # correct
    _set(act["pere"]["age"], "trente-un", 31, 0.6)  # faux
    _set(act["mere"]["nom"], None, None, None)  # manquant
    _set(act["mere"]["profession"], "ménagère", "ménagère", 0.4)  # en trop
    _set(act["temoins"][0]["nom"], "Invente", "Invente", 0.8)  # faux (accent)
    return act


def _leaves(act: dict) -> list:
    return list(iter_leaves(act))


def test_iter_leaves_paths() -> None:
    paths = [path for path, _ in _leaves(MODELE)]
    assert "numero_acte" in paths
    assert "enfant.date_naissance" in paths
    assert "temoins[1].profession" in paths
    assert "controle.statut" not in paths
    assert len(paths) == 2 + 4 + 4 * 4


def test_extract_fields_prefers_normalise_then_brut() -> None:
    act = copy.deepcopy(MODELE)
    _set(act["enfant"]["nom"], "Fictivard", "FICTIVARD", 0.7)
    _set(act["enfant"]["prenoms"], "Paul", None)
    _set(act["pere"]["age"], "trente", 30)
    _set(act["mere"]["nom"], "  ", None)
    fields = extract_fields(act)
    assert fields["enfant.nom"].value == "FICTIVARD"
    assert fields["enfant.nom"].confidence == 0.7
    assert fields["enfant.prenoms"].value == "Paul"
    assert fields["pere.age"].value == "30"
    assert fields["mere.nom"].value is None  # vide = absent


def test_field_group() -> None:
    assert field_group("temoins[12].nom") == "temoins.nom"
    assert field_group("enfant.nom") == "enfant.nom"


@pytest.mark.parametrize(
    ("ref", "hyp", "outcome"),
    [
        ("a", "a", Outcome.CORRECT),
        ("a", "b", Outcome.WRONG),
        ("a", None, Outcome.MISSING),
        (None, "b", Outcome.EXTRA),
        (None, None, Outcome.ABSENT),
    ],
)
def test_classify(ref, hyp, outcome) -> None:
    assert classify(ref, hyp) is outcome


def test_field_stats() -> None:
    stats = FieldStats()
    assert stats.precision is None and stats.recall is None and stats.f1 is None
    for outcome in [Outcome.CORRECT, Outcome.CORRECT, Outcome.WRONG, Outcome.MISSING]:
        stats.add(outcome)
    stats.add(Outcome.EXTRA)
    stats.add(Outcome.ABSENT)
    assert (stats.true_positives, stats.false_positives, stats.false_negatives) == (2, 2, 2)
    assert stats.support == 4
    assert stats.precision == pytest.approx(0.5)
    assert stats.recall == pytest.approx(0.5)
    assert stats.f1 == pytest.approx(0.5)


def test_f1_zero_when_nothing_correct() -> None:
    stats = FieldStats()
    stats.add(Outcome.WRONG)
    assert stats.f1 == 0.0


def test_compare_acts(reference: dict, hypothesis: dict) -> None:
    outcomes = {c.path: c.outcome for c in compare_acts(reference, hypothesis)}
    assert outcomes["enfant.prenoms"] is Outcome.CORRECT
    assert outcomes["pere.age"] is Outcome.WRONG
    assert outcomes["mere.nom"] is Outcome.MISSING
    assert outcomes["mere.profession"] is Outcome.EXTRA
    assert outcomes["temoins[0].nom"] is Outcome.WRONG
    assert outcomes["temoins[1].nom"] is Outcome.ABSENT


def test_compare_acts_diacritics_normalization(reference: dict, hypothesis: dict) -> None:
    options = Normalization(diacritics=True)
    outcomes = {c.path: c.outcome for c in compare_acts(reference, hypothesis, options)}
    assert outcomes["temoins[0].nom"] is Outcome.CORRECT


def test_compare_acts_nfd_equals_nfc(reference: dict) -> None:
    hyp = copy.deepcopy(reference)
    _set(hyp["temoins"][0]["nom"], "Inventé", "Inventé")  # NFD
    outcomes = {c.path: c.outcome for c in compare_acts(reference, hyp)}
    assert outcomes["temoins[0].nom"] is Outcome.CORRECT


def test_compare_acts_different_witness_counts(reference: dict) -> None:
    hyp = copy.deepcopy(reference)
    hyp["temoins"] = hyp["temoins"][:1]
    extra = copy.deepcopy(reference["temoins"][0])
    _set(extra["nom"], "Supposé", "Supposé")
    hyp["temoins"] += [copy.deepcopy(MODELE["temoins"][0]), extra]
    outcomes = {c.path: c.outcome for c in compare_acts(reference, hyp)}
    assert outcomes["temoins[2].nom"] is Outcome.EXTRA


def test_missing_hypothesis_act(reference: dict) -> None:
    comparisons = compare_acts(reference, None)
    present = [c for c in comparisons if c.reference is not None]
    assert present and all(c.outcome is Outcome.MISSING for c in present)


def test_evaluate_acts(reference: dict, hypothesis: dict) -> None:
    other = copy.deepcopy(reference)
    result = evaluate_acts(
        {"a": reference, "b": other}, {"a": hypothesis, "c": hypothesis}, Normalization()
    )
    assert result.missing_acts == ["b"]
    assert result.extra_acts == ["c"]
    stats = result.by_field()
    assert stats["pere.age"].precision == 0.0
    assert stats["pere.age"].recall == 0.0
    assert stats["enfant.nom"].precision == 1.0
    assert stats["enfant.nom"].recall == pytest.approx(0.5)  # acte b manquant
    assert stats["temoins.nom"].support == 2
    overall = result.overall()
    # Acte a : 8 valeurs de référence, 5 correctes, 2 fausses, 1 manquante, 1 en trop.
    # Acte b : 8 manquantes.
    assert overall.true_positives == 5
    assert overall.false_positives == 3
    assert overall.false_negatives == 2 + 1 + 8
    data = result.to_dict()
    assert data["normalisation"] == ["nfc"]
    assert {e["issue"] for e in data["ecarts"]} == {"faux", "manquant", "en_trop"}


def test_confidence_pairs(reference: dict, hypothesis: dict) -> None:
    pairs = evaluate_acts({"a": reference}, {"a": hypothesis}).confidence_pairs()
    assert (0.95, True) in pairs
    assert (0.6, False) in pairs
    assert (0.4, False) in pairs  # valeur en trop
    assert len(pairs) == 8  # 7 valeurs produites présentes en référence + 1 en trop


def test_load_acts_by_id_or_filename(tmp_path: Path, reference: dict) -> None:
    (tmp_path / "x.json").write_text(json.dumps(reference), "utf-8")
    assert list(load_acts(tmp_path)) == ["fictif-N-1893-001"]
    (tmp_path / "y.json").write_text(json.dumps(MODELE), "utf-8")  # id null
    assert list(load_acts(tmp_path)) == ["x", "y"]


def test_load_acts_errors(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{", "utf-8")
    (tmp_path / "list.json").write_text("[]", "utf-8")
    for name in ["bad.json", "list.json", "absent.json"]:
        with pytest.raises(ActError):
            load_acts(tmp_path / name)


def test_evaluate_paths(tmp_path: Path, reference: dict, hypothesis: dict) -> None:
    ref_file, hyp_file = tmp_path / "ref.json", tmp_path / "hyp.json"
    ref_file.write_text(json.dumps(reference), "utf-8")
    hypothesis["id"] = "autre-id-fictif"
    hyp_file.write_text(json.dumps(hypothesis), "utf-8")
    result = evaluate_paths(ref_file, hyp_file)
    assert result.missing_acts == []
    assert result.overall().true_positives == 5
    with pytest.raises(ActError):
        evaluate_paths(ref_file, tmp_path)
