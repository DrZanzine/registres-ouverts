# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lecture des transcriptions et comparaison page par page.

Deux formats sont acceptés, pour la référence comme pour l'hypothèse :

- texte brut (``.txt``) : une ligne du fichier = une ligne de l'image, sans
  identifiant ;
- PAGE XML (``.xml``, toutes versions, dont 2019 produite par kraken) : le texte
  de chaque ``TextLine`` est lu dans son ``TextEquiv/Unicode`` direct (pas
  celui des ``Word``), dans l'ordre du document.

Appariement des lignes : par identifiant quand les deux côtés en ont
(PAGE XML), sinon par ordre. Une ligne de référence sans correspondant compte
comme entièrement supprimée, une ligne d'hypothèse en trop comme entièrement
insérée ; les deux sont signalées.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from registres_ouverts.evaluate.metrics import Score, TextComparison, char_confusions, compare_text
from registres_ouverts.evaluate.normalize import Normalization

TEXT_SUFFIXES = {".txt"}
PAGE_SUFFIXES = {".xml"}


class TranscriptionError(ValueError):
    """Fichier de transcription illisible ou invalide."""


@dataclass(frozen=True, slots=True)
class Line:
    """Une ligne transcrite ; ``id`` vaut ``None`` pour un fichier texte."""

    text: str
    id: str | None = None


def _local(tag: str) -> str:
    """Nom local d'un élément XML, sans espace de noms."""
    return tag.rsplit("}", 1)[-1]


def _textequiv_text(line: ET.Element) -> str:
    """Texte du ``TextEquiv`` direct d'une ``TextLine``.

    S'il y en a plusieurs, celui d'``index`` le plus petit est retenu (le
    premier rencontré à défaut d'index).
    """
    candidates: list[tuple[int, int, str]] = []
    for position, child in enumerate(line):
        if _local(child.tag) != "TextEquiv":
            continue
        unicode = next((e for e in child if _local(e.tag) == "Unicode"), None)
        text = unicode.text if unicode is not None and unicode.text else ""
        try:
            index = int(child.get("index", position))
        except ValueError:
            index = position
        candidates.append((index, position, text))
    if not candidates:
        return ""
    return min(candidates)[2]


def parse_page_xml(content: str | bytes) -> list[Line]:
    """Lit les lignes d'un document PAGE XML."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        raise TranscriptionError(f"PAGE XML invalide : {error}") from error
    if _local(root.tag) != "PcGts":
        raise TranscriptionError(f"racine PcGts attendue, trouvé {_local(root.tag)!r}")
    return [
        Line(text=_textequiv_text(element), id=element.get("id"))
        for element in root.iter()
        if _local(element.tag) == "TextLine"
    ]


def parse_text(content: str) -> list[Line]:
    """Lit un fichier texte : une ligne par ligne, fin de fichier ignorée."""
    return [Line(text=text) for text in content.splitlines()]


def load_lines(path: Path) -> list[Line]:
    """Charge une transcription selon son extension (``.txt`` ou ``.xml``)."""
    suffix = path.suffix.lower()
    try:
        if suffix in PAGE_SUFFIXES:
            return parse_page_xml(path.read_bytes())
        if suffix in TEXT_SUFFIXES:
            return parse_text(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise TranscriptionError(f"{path} : {error}") from error
    except UnicodeDecodeError as error:
        raise TranscriptionError(f"{path} : encodage UTF-8 attendu ({error})") from error
    except TranscriptionError as error:
        raise TranscriptionError(f"{path} : {error}") from error
    raise TranscriptionError(f"{path} : extension non reconnue (.txt ou .xml attendu)")


class LineStatus(StrEnum):
    """Statut d'appariement d'une ligne."""

    MATCHED = "appariee"
    MISSING = "manquante"  # dans la référence, absente de l'hypothèse
    EXTRA = "en_trop"  # dans l'hypothèse, absente de la référence


@dataclass(slots=True)
class LineResult:
    """Résultat d'une ligne appariée (ou non)."""

    status: LineStatus
    reference_id: str | None
    hypothesis_id: str | None
    reference_index: int | None
    hypothesis_index: int | None
    comparison: TextComparison

    def to_dict(self) -> dict:
        """Représentation JSON."""
        return {
            "statut": str(self.status),
            "id_reference": self.reference_id,
            "id_hypothese": self.hypothesis_id,
            "rang_reference": self.reference_index,
            "rang_hypothese": self.hypothesis_index,
            "reference": self.comparison.reference,
            "hypothese": self.comparison.hypothesis,
            "cer": self.comparison.cer.to_dict(),
            "wer": self.comparison.wer.to_dict(),
        }


@dataclass(slots=True)
class PageResult:
    """Résultat d'une page : lignes et totaux (micro)."""

    name: str
    pairing: str  # "identifiant" ou "ordre"
    lines: list[LineResult] = field(default_factory=list)

    @property
    def cer(self) -> Score:
        """CER global de la page."""
        return sum((line.comparison.cer for line in self.lines), Score())

    @property
    def wer(self) -> Score:
        """WER global de la page."""
        return sum((line.comparison.wer for line in self.lines), Score())

    def count(self, status: LineStatus) -> int:
        """Nombre de lignes ayant ce statut."""
        return sum(1 for line in self.lines if line.status is status)

    def to_dict(self) -> dict:
        """Représentation JSON."""
        return {
            "page": self.name,
            "appariement": self.pairing,
            "lignes_appariees": self.count(LineStatus.MATCHED),
            "lignes_manquantes": self.count(LineStatus.MISSING),
            "lignes_en_trop": self.count(LineStatus.EXTRA),
            "cer": self.cer.to_dict(),
            "wer": self.wer.to_dict(),
            "lignes": [line.to_dict() for line in self.lines],
        }


def _has_unique_ids(lines: list[Line]) -> bool:
    ids = [line.id for line in lines]
    return all(ids) and len(set(ids)) == len(ids)


def pair_lines(
    reference: list[Line], hypothesis: list[Line], by_order: bool = False
) -> tuple[str, list[tuple[int | None, int | None]]]:
    """Apparie les lignes ; renvoie le mode retenu et les couples d'indices.

    Par identifiant si ``by_order`` est faux et que chaque côté a des
    identifiants uniques (et non vides) ; sinon par ordre. Un indice ``None``
    marque une ligne sans correspondant. L'ordre de sortie suit la référence,
    les lignes en trop étant placées à la fin.
    """
    use_ids = not by_order and bool(reference) and bool(hypothesis)
    if use_ids and _has_unique_ids(reference) and _has_unique_ids(hypothesis):
        hyp_index = {line.id: i for i, line in enumerate(hypothesis)}
        pairs: list[tuple[int | None, int | None]] = [
            (i, hyp_index.get(line.id)) for i, line in enumerate(reference)
        ]
        used = {j for _, j in pairs if j is not None}
        pairs += [(None, j) for j in range(len(hypothesis)) if j not in used]
        return "identifiant", pairs
    common = min(len(reference), len(hypothesis))
    pairs = [(i, i) for i in range(common)]
    pairs += [(i, None) for i in range(common, len(reference))]
    pairs += [(None, j) for j in range(common, len(hypothesis))]
    return "ordre", pairs


def compare_lines(
    reference: list[Line],
    hypothesis: list[Line],
    normalization: Normalization | None = None,
    by_order: bool = False,
    name: str = "",
) -> PageResult:
    """Compare deux transcriptions d'une même page, ligne à ligne."""
    pairing, pairs = pair_lines(reference, hypothesis, by_order)
    page = PageResult(name=name, pairing=pairing)
    for i, j in pairs:
        ref = reference[i] if i is not None else None
        hyp = hypothesis[j] if j is not None else None
        if ref is None:
            status = LineStatus.EXTRA
        elif hyp is None:
            status = LineStatus.MISSING
        else:
            status = LineStatus.MATCHED
        page.lines.append(
            LineResult(
                status=status,
                reference_id=ref.id if ref else None,
                hypothesis_id=hyp.id if hyp else None,
                reference_index=i,
                hypothesis_index=j,
                comparison=compare_text(
                    ref.text if ref else "", hyp.text if hyp else "", normalization
                ),
            )
        )
    return page


def compare_files(
    reference: Path,
    hypothesis: Path,
    normalization: Normalization | None = None,
    by_order: bool = False,
) -> PageResult:
    """Compare deux fichiers de transcription (``.txt`` ou PAGE XML)."""
    return compare_lines(
        load_lines(reference),
        load_lines(hypothesis),
        normalization,
        by_order,
        name=reference.stem,
    )


def _transcription_files(directory: Path) -> dict[str, Path]:
    """Fichiers de transcription d'un dossier, indexés par nom sans extension."""
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES | PAGE_SUFFIXES:
            if path.stem in files:
                raise TranscriptionError(f"{directory} : deux fichiers pour la page {path.stem!r}")
            files[path.stem] = path
    return files


@dataclass(slots=True)
class CorpusResult:
    """Résultat sur plusieurs pages."""

    normalization: Normalization
    pages: list[PageResult] = field(default_factory=list)
    missing_pages: list[str] = field(default_factory=list)  # référence sans hypothèse
    extra_pages: list[str] = field(default_factory=list)  # hypothèse sans référence

    @property
    def cer(self) -> Score:
        """CER global (micro) sur toutes les pages."""
        return sum((page.cer for page in self.pages), Score())

    @property
    def wer(self) -> Score:
        """WER global (micro) sur toutes les pages."""
        return sum((page.wer for page in self.pages), Score())

    def confusions(self) -> Counter[tuple[str, str]]:
        """Confusions de caractères cumulées sur les lignes appariées.

        Les lignes manquantes ou en trop sont exclues : ce ne sont pas des
        confusions de lecture mais des erreurs de segmentation.
        """
        return char_confusions(
            line.comparison
            for page in self.pages
            for line in page.lines
            if line.status is LineStatus.MATCHED
        )

    def to_dict(self, top_confusions: int = 20) -> dict:
        """Représentation JSON."""
        return {
            "normalisation": self.normalization.labels(),
            "pages": len(self.pages),
            "pages_manquantes": self.missing_pages,
            "pages_en_trop": self.extra_pages,
            "lignes_manquantes": sum(p.count(LineStatus.MISSING) for p in self.pages),
            "lignes_en_trop": sum(p.count(LineStatus.EXTRA) for p in self.pages),
            "cer": self.cer.to_dict(),
            "wer": self.wer.to_dict(),
            "confusions": [
                {"reference": r, "hypothese": h, "nombre": n}
                for (r, h), n in self.confusions().most_common(top_confusions)
            ],
            "detail": [page.to_dict() for page in self.pages],
        }


def compare_paths(
    reference: Path,
    hypothesis: Path,
    normalization: Normalization | None = None,
    by_order: bool = False,
) -> CorpusResult:
    """Compare deux fichiers, ou deux dossiers page par page.

    Dans des dossiers, les pages sont appariées par nom de fichier sans
    extension (``vue_0002.txt`` ↔ ``vue_0002.xml``). Une page de référence
    sans hypothèse est signalée et ses lignes comptées comme supprimées ; une
    page d'hypothèse sans référence est signalée mais non comptée.
    """
    normalization = normalization or Normalization()
    result = CorpusResult(normalization=normalization)
    if reference.is_dir() != hypothesis.is_dir():
        raise TranscriptionError("référence et hypothèse : deux fichiers ou deux dossiers")
    if not reference.is_dir():
        result.pages.append(compare_files(reference, hypothesis, normalization, by_order))
        return result
    refs, hyps = _transcription_files(reference), _transcription_files(hypothesis)
    for name, ref_path in refs.items():
        if name in hyps:
            result.pages.append(compare_files(ref_path, hyps[name], normalization, by_order))
        else:
            result.missing_pages.append(name)
            result.pages.append(
                compare_lines(load_lines(ref_path), [], normalization, by_order, name=name)
            )
    result.extra_pages = [name for name in hyps if name not in refs]
    return result
