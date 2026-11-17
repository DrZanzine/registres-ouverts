# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests du manifeste des images brutes.

Toutes les images et références utilisées ici sont FICTIVES : de simples
octets écrits dans un dossier temporaire.
"""

import csv
import hashlib
from pathlib import Path

from registres_ouverts.acquire.manifest import (
    COLUMNS,
    generate,
    main,
    parse_filename,
)

FAKE_NAME = "AD99_9ZZ9_0001_0000_0042.jpg"  # fictif


def _make_image(directory: Path, name: str, content: bytes = b"image fictive") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_parse_filename_ad04_format() -> None:
    assert parse_filename("AD04_1MI5_1565_1144_0002.jpg") == {
        "archives": "AD04",
        "cote": "1MI5/1565",
        "vue": "2",
    }


def test_parse_filename_unknown_format_returns_nothing() -> None:
    assert parse_filename("scan 12.jpg") == {}
    assert parse_filename("AD04_1MI5_1565_0002.jpg") == {}


def test_manifest_columns_and_values(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    image = _make_image(raw / "AD99" / "9ZZ9_0001", FAKE_NAME)
    generate(raw)

    rows = _read(raw / "manifest.csv")
    assert list(rows[0].keys()) == COLUMNS
    assert COLUMNS == [
        "fichier",
        "archives",
        "cote",
        "vue",
        "commune",
        "type_acte",
        "annee",
        "sha256",
    ]
    row = rows[0]
    assert row["fichier"] == f"AD99/9ZZ9_0001/{FAKE_NAME}"
    assert row["archives"] == "AD99"
    assert row["cote"] == "9ZZ9/0001"
    assert row["vue"] == "42"
    assert row["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    # Non déductibles du nom : restent vides.
    assert row["commune"] == row["type_acte"] == row["annee"] == ""


def test_unrecognised_file_is_listed_with_empty_references(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_image(raw, "page sans nom.png")
    _make_image(raw, "notes.txt")  # pas une image : ignoré

    rows = generate(raw)
    assert [r.fichier for r in rows] == ["page sans nom.png"]
    assert rows[0].archives == rows[0].cote == rows[0].vue == ""
    assert rows[0].sha256


def test_regeneration_keeps_manual_values(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_image(raw, FAKE_NAME)
    manifest = raw / "manifest.csv"
    generate(raw)

    rows = _read(manifest)
    rows[0].update(commune="Villefictive", type_acte="naissance", annee="1900")
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    _make_image(raw, "AD99_9ZZ9_0001_0000_0043.jpg", b"autre image fictive")
    generate(raw)

    by_file = {r["fichier"]: r for r in _read(manifest)}
    assert by_file[FAKE_NAME]["commune"] == "Villefictive"
    assert by_file[FAKE_NAME]["type_acte"] == "naissance"
    assert by_file[FAKE_NAME]["annee"] == "1900"
    assert by_file["AD99_9ZZ9_0001_0000_0043.jpg"]["commune"] == ""


def test_manifest_itself_is_not_listed(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_image(raw, FAKE_NAME)
    generate(raw)
    rows = generate(raw)
    assert [r.fichier for r in rows] == [FAKE_NAME]


def test_cli(tmp_path: Path, capsys) -> None:
    raw = tmp_path / "raw"
    _make_image(raw, FAKE_NAME)
    assert main(["--raw-dir", str(raw)]) == 0
    assert (raw / "manifest.csv").exists()
    assert "1 image(s)" in capsys.readouterr().out
