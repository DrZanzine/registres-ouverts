# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validation du schéma d'un acte et du modèle de référence vide.

Les valeurs de l'acte rempli ci-dessous sont FICTIVES.
"""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs" / "acte.schema.json"
MODELE_PATH = ROOT / "data" / "reference" / "MODELE.json"


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.fixture
def modele() -> dict:
    return json.loads(MODELE_PATH.read_text(encoding="utf-8"))


def _field(brut, normalise, confiance=None) -> dict:
    return {"brut": brut, "normalise": normalise, "confiance": confiance}


def test_modele_is_valid(validator: Draft202012Validator, modele: dict) -> None:
    validator.validate(modele)


def test_modele_is_empty(modele: dict) -> None:
    """Le modèle ne contient aucune valeur lue : tout champ texte est vide."""

    def leaves(node):
        if isinstance(node, dict):
            if set(node) == {"brut", "normalise", "confiance"}:
                yield node
            else:
                for value in node.values():
                    yield from leaves(value)
        elif isinstance(node, list):
            for value in node:
                yield from leaves(value)

    fields = list(leaves(modele))
    assert fields
    assert all(v is None for f in fields for v in f.values())


def test_filled_act_is_valid(validator: Draft202012Validator, modele: dict) -> None:
    acte = copy.deepcopy(modele)  # valeurs fictives
    acte["source"] = {"archives": "AD99", "cote": "9ZZ9/0001", "vue": 3, "zone": [0, 0, 10, 10]}
    acte["numero_acte"] = _field("un", 1, 0.9)
    acte["date_acte"] = _field("premier janvier", "1900-01-01")
    acte["enfant"]["nom"] = _field("Fictif", "FICTIF", 0.8)
    acte["enfant"]["sexe"] = _field("masculin", "M")
    acte["enfant"]["date_naissance"] = _field("la veille à dix heures", "1899-12-31T10:00")
    acte["pere"]["age"] = _field("trente ans", 30)
    validator.validate(acte)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("enfant", "nom"), {"normalise": "X", "confiance": None}),  # brut manquant
        (("date_acte",), _field("x", "01/01/1900")),  # format de date
        (("pere", "age"), _field("trente", "30")),  # âge non entier
        (("enfant", "sexe"), _field("x", "H")),  # sexe hors liste
        (("controle", "statut"), "ok"),  # statut hors liste
        (("source", "zone"), [1, 2, 3]),  # zone incomplète
    ],
)
def test_invalid_acts_are_rejected(
    validator: Draft202012Validator, modele: dict, path: tuple, value
) -> None:
    acte = copy.deepcopy(modele)
    target = acte
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        validator.validate(acte)


def test_unknown_field_is_rejected(validator: Draft202012Validator, modele: dict) -> None:
    acte = copy.deepcopy(modele)
    acte["enfant"]["surnom"] = _field(None, None)
    with pytest.raises(ValidationError):
        validator.validate(acte)
