# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Interface en ligne de commande du module d'évaluation.

Toutes les données sont FICTIVES (Fictivard, Imaginaire, Fictiville).
"""

import copy
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from registres_ouverts.evaluate.cli import EXIT_ERROR, EXIT_FAILED, EXIT_OK, main

ROOT = Path(__file__).resolve().parents[1]
MODELE = json.loads((ROOT / "data" / "reference" / "MODELE.json").read_text(encoding="utf-8"))


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main([str(a) for a in argv], out)
    return code, out.getvalue()


@pytest.fixture
def pages(tmp_path: Path) -> tuple[Path, Path]:
    ref, hyp = tmp_path / "ref", tmp_path / "hyp"
    ref.mkdir()
    hyp.mkdir()
    (ref / "vue_0002.txt").write_text(
        "Jean Fictivard, cultivateur fictif\nné à Fictiville\nde Marie Imaginaire\n", "utf-8"
    )
    (hyp / "vue_0002.txt").write_text(
        "Jean Fictivard cultivateur fictif\nne a Fictiville\n", "utf-8"
    )
    return ref, hyp


@pytest.fixture
def acts(tmp_path: Path) -> tuple[Path, Path]:
    ref = copy.deepcopy(MODELE)
    ref["id"] = "fictif-1"
    ref["enfant"]["nom"].update(brut="Fictivard", normalise="Fictivard")
    ref["pere"]["nom"].update(brut="Fictivard", normalise="Fictivard")
    ref["mere"]["nom"].update(brut="Imaginaire", normalise="Imaginaire")
    hyp = copy.deepcopy(ref)
    hyp["enfant"]["nom"].update(confiance=0.95)
    hyp["pere"]["nom"].update(confiance=0.9)
    hyp["mere"]["nom"].update(brut="Imaginair", normalise="Imaginair", confiance=0.5)
    ref_path, hyp_path = tmp_path / "ref.json", tmp_path / "hyp.json"
    ref_path.write_text(json.dumps(ref), "utf-8")
    hyp_path.write_text(json.dumps(hyp), "utf-8")
    return ref_path, hyp_path


# --- cer -----------------------------------------------------------------------


def test_cer_text_output(pages: tuple[Path, Path]) -> None:
    code, out = run("cer", *pages, "--lignes")
    assert code == EXIT_OK
    assert "Normalisation : nfc" in out
    assert "CER : " in out and "WER : " in out
    assert "ligne manquante dans l'hypothèse : n° 3" in out
    assert "«é» → «e»" in out


def test_cer_json_output(pages: tuple[Path, Path]) -> None:
    code, out = run("cer", *pages, "--json", "--casse", "--diacritiques")
    assert code == EXIT_OK
    data = json.loads(out)
    assert data["normalisation"] == ["nfc", "casse", "diacritiques"]
    assert data["lignes_manquantes"] == 1
    assert data["detail"][0]["page"] == "vue_0002"
    assert data["cer"]["erreurs"] == 1 + len("de Marie Imaginaire")


def test_cer_max_threshold(pages: tuple[Path, Path], tmp_path: Path) -> None:
    code, out = run("cer", *pages, "--max", "0.05")
    assert code == EXIT_FAILED
    assert "supérieur au maximum" in out
    same = tmp_path / "same.txt"
    same.write_text("Jean Fictivard\n", "utf-8")
    assert run("cer", same, same, "--max", "0")[0] == EXIT_OK


def test_cer_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = run("cer", tmp_path / "absent.txt", tmp_path / "absent2.txt")
    assert code == EXIT_ERROR
    assert "erreur" in capsys.readouterr().err


def test_usage_errors() -> None:
    assert run()[0] == EXIT_ERROR
    assert run("inconnu")[0] == EXIT_ERROR
    assert run("cer", "a.txt", "b.txt", "--max", "2")[0] == EXIT_ERROR
    assert run("--help")[0] == EXIT_OK


# --- champs ----------------------------------------------------------------------


def test_champs_text_output(acts: tuple[Path, Path]) -> None:
    code, out = run("champs", *acts)
    assert code == EXIT_OK
    assert "enfant.nom" in out
    assert "global" in out
    assert "Écarts (1)" in out
    assert "'Imaginair'" in out


def test_champs_json_and_export(acts: tuple[Path, Path], tmp_path: Path) -> None:
    export = tmp_path / "confiances.csv"
    code, out = run("champs", *acts, "--json", "--export-confiances", export)
    assert code == EXIT_OK
    data = json.loads(out)
    assert data["champs"]["mere.nom"]["precision"] == 0.0
    assert data["champs"]["enfant.nom"]["f1"] == 1.0
    assert data["global"]["vrais_positifs"] == 2
    assert export.read_text("utf-8").splitlines() == [
        "confiance,correct",
        "0.95,1",
        "0.9,1",
        "0.5,0",
    ]


def test_champs_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{", "utf-8")
    assert run("champs", bad, bad)[0] == EXIT_ERROR


# --- seuil -----------------------------------------------------------------------


def test_seuil_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    path.write_text("confiance,correct\n0.95,1\n0.9,1\n0.8,0\n0.7,1\n", "utf-8")
    code, out = run("seuil", path, "--cible", "0.99")
    assert code == EXIT_OK
    assert "Seuil retenu pour 99.00 % de précision : 0.900" in out
    code, out = run("seuil", path, "--json", "--pas", "0.5")
    data = json.loads(out)
    assert [p["seuil"] for p in data["courbe"]] == [0.0, 0.5, 1.0]
    assert data["seuil_retenu"] is None  # précision 75 % à 0 et 0.5, rien à 1


def test_seuil_unreachable(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    path.write_text("confiance,correct\n0.9,0\n", "utf-8")
    code, out = run("seuil", path)
    assert code == EXIT_OK
    assert "Aucun seuil" in out


def test_seuil_from_acts(acts: tuple[Path, Path]) -> None:
    code, out = run("seuil", "--actes", *acts, "--json")
    assert code == EXIT_OK
    data = json.loads(out)
    assert data["valeurs"] == 3
    assert data["seuil_retenu"]["seuil"] == 0.9


def test_seuil_errors(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    path.write_text("confiance,correct\n0.9,1\n", "utf-8")
    assert run("seuil", path, "--pas", "0")[0] == EXIT_ERROR
    assert run("seuil")[0] == EXIT_ERROR
    assert run("seuil", tmp_path / "absent.csv")[0] == EXIT_ERROR


def test_module_entry_point(pages: tuple[Path, Path]) -> None:
    """``python -m registres_ouverts.evaluate`` fonctionne et propage le code."""
    result = subprocess.run(
        [sys.executable, "-m", "registres_ouverts.evaluate", "cer", *map(str, pages), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["pages"] == 1
    result = subprocess.run(
        [sys.executable, "-m", "registres_ouverts.evaluate", "cer", "x.txt"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
