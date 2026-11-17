# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Génère le manifeste des images brutes (data/raw/manifest.csv).

Une ligne par image : chemin relatif, références d'archives déduites du nom
de fichier quand c'est possible, et empreinte SHA-256. Ce qui ne peut pas être
déduit reste vide, à compléter à la main. Lors d'une régénération, les valeurs
saisies à la main dans un manifeste existant sont conservées.

Nom de fichier attendu (Archives départementales des Alpes-de-Haute-Provence) :
``AD04_1MI5_1565_1144_0002.jpg`` → archives ``AD04``, cote ``1MI5/1565``,
vue ``2``. Le quatrième segment (``1144``) n'est pas interprété.

Usage : ``python -m registres_ouverts.acquire.manifest [--raw-dir data/raw]``
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

FILENAME_PATTERN = re.compile(
    r"^(?P<archives>AD\d{2}[AB]?|AD\d{3})"
    r"_(?P<series>[0-9A-Za-z]+)"
    r"_(?P<item>[0-9A-Za-z]+)"
    r"_(?P<unknown>\d+)"
    r"_(?P<view>\d+)$"
)

DEFAULT_RAW_DIR = Path("data/raw")
MANIFEST_NAME = "manifest.csv"


@dataclass
class ManifestRow:
    """Une ligne du manifeste. Les noms de champs sont les colonnes du CSV."""

    fichier: str
    archives: str = ""
    cote: str = ""
    vue: str = ""
    commune: str = ""
    type_acte: str = ""
    annee: str = ""
    sha256: str = ""


COLUMNS = [f.name for f in fields(ManifestRow)]


def parse_filename(name: str) -> dict[str, str]:
    """Déduit archives, cote et vue d'un nom de fichier.

    Renvoie un dictionnaire vide si le nom ne suit pas le format attendu :
    rien n'est deviné.
    """
    match = FILENAME_PATTERN.match(Path(name).stem)
    if match is None:
        return {}
    return {
        "archives": match["archives"].upper(),
        "cote": f"{match['series']}/{match['item']}",
        "vue": str(int(match["view"])),
    }


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def find_images(raw_dir: Path) -> list[Path]:
    """Liste récursivement les images de ``raw_dir``, triées par chemin."""
    return sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Lit un manifeste existant, indexé par la colonne ``fichier``."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["fichier"]: row for row in csv.DictReader(handle) if row.get("fichier")}


def build_rows(
    raw_dir: Path, previous: dict[str, dict[str, str]] | None = None
) -> list[ManifestRow]:
    """Construit les lignes du manifeste pour toutes les images de ``raw_dir``.

    Une valeur déduite du nom de fichier ou calculée (sha256) prime ; une
    valeur vide est complétée par celle du manifeste précédent, s'il y en a une.
    """
    previous = previous or {}
    rows = []
    for image in find_images(raw_dir):
        relative = image.relative_to(raw_dir).as_posix()
        row = ManifestRow(fichier=relative, sha256=sha256_of(image), **parse_filename(image.name))
        old = previous.get(relative, {})
        for column in COLUMNS:
            if not getattr(row, column) and old.get(column):
                setattr(row, column, old[column])
        rows.append(row)
    return rows


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    """Écrit le manifeste en CSV UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def generate(raw_dir: Path, output: Path | None = None) -> list[ManifestRow]:
    """Génère (ou régénère) le manifeste de ``raw_dir`` et renvoie ses lignes."""
    output = output or raw_dir / MANIFEST_NAME
    rows = build_rows(raw_dir, read_manifest(output))
    write_manifest(rows, output)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=None, help="défaut : <raw-dir>/manifest.csv")
    args = parser.parse_args(argv)

    if not args.raw_dir.is_dir():
        parser.error(f"dossier introuvable : {args.raw_dir}")
    rows = generate(args.raw_dir, args.output)
    unparsed = [r.fichier for r in rows if not r.cote]
    print(f"{len(rows)} image(s) dans le manifeste.")
    for name in unparsed:
        print(f"  nom non reconnu, références à compléter : {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
