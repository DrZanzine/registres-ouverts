# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lecture texte / PAGE XML et comparaison page par page.

Tous les textes sont FICTIFS (Fictivard, Imaginaire, Fictiville).
"""

import unicodedata
from pathlib import Path

import pytest

from registres_ouverts.evaluate.normalize import Normalization
from registres_ouverts.evaluate.transcription import (
    Line,
    LineStatus,
    TranscriptionError,
    compare_files,
    compare_lines,
    compare_paths,
    load_lines,
    pair_lines,
    parse_page_xml,
    parse_text,
)

NS_2019 = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
NS_2013 = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def page_xml(lines: list[tuple[str, str]], ns: str = NS_2019) -> str:
    """Construit un PAGE XML minimal, au format produit par kraken."""
    body = "".join(
        f'<TextLine id="{line_id}"><Coords points="0,0 1,1"/>'
        f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>"
        for line_id, text in lines
    )
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><PcGts xmlns="{ns}">'
        f"<Metadata><Creator>fictif</Creator></Metadata>"
        f'<Page imageFilename="fictif.jpg" imageWidth="10" imageHeight="10">'
        f'<TextRegion id="r1">{body}</TextRegion></Page></PcGts>'
    )


# --- Lecture ---------------------------------------------------------------


def test_parse_text() -> None:
    assert parse_text("Jean Fictivard\nné le deux mai\n") == [
        Line("Jean Fictivard"),
        Line("né le deux mai"),
    ]
    assert parse_text("") == []
    assert parse_text("a\n\nb") == [Line("a"), Line(""), Line("b")]


@pytest.mark.parametrize("ns", [NS_2019, NS_2013])
def test_parse_page_xml(ns: str) -> None:
    xml = page_xml([("l1", "Jean Fictivard"), ("l2", "Fictiville")], ns)
    assert parse_page_xml(xml) == [Line("Jean Fictivard", "l1"), Line("Fictiville", "l2")]


def test_parse_page_xml_ignores_word_textequiv_and_picks_lowest_index() -> None:
    xml = f"""<PcGts xmlns="{NS_2019}"><Page><TextRegion id="r">
      <TextLine id="a">
        <Word id="w"><TextEquiv><Unicode>MOT</Unicode></TextEquiv></Word>
        <TextEquiv index="2"><Unicode>second</Unicode></TextEquiv>
        <TextEquiv index="1"><Unicode>premier</Unicode></TextEquiv>
      </TextLine>
      <TextLine id="b"><Coords points="0,0"/></TextLine>
      <TextLine id="c"><TextEquiv><Unicode/></TextEquiv></TextLine>
    </TextRegion></Page></PcGts>"""
    assert parse_page_xml(xml) == [Line("premier", "a"), Line("", "b"), Line("", "c")]


def test_parse_page_xml_errors() -> None:
    with pytest.raises(TranscriptionError):
        parse_page_xml("<PcGts><oups>")
    with pytest.raises(TranscriptionError):
        parse_page_xml("<autre/>")


def test_load_lines(tmp_path: Path) -> None:
    (tmp_path / "p.txt").write_text("une\ndeux\n", encoding="utf-8")
    (tmp_path / "p.xml").write_text(page_xml([("l1", "une")]), encoding="utf-8")
    (tmp_path / "p.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "latin1.txt").write_bytes("né".encode("latin-1"))
    assert len(load_lines(tmp_path / "p.txt")) == 2
    assert load_lines(tmp_path / "p.xml") == [Line("une", "l1")]
    for name in ["p.pdf", "absent.txt", "latin1.txt"]:
        with pytest.raises(TranscriptionError):
            load_lines(tmp_path / name)


# --- Appariement -----------------------------------------------------------


def test_pair_by_order_with_missing_and_extra() -> None:
    ref = [Line("a"), Line("b"), Line("c")]
    mode, pairs = pair_lines(ref, [Line("a")])
    assert mode == "ordre"
    assert pairs == [(0, 0), (1, None), (2, None)]
    _, pairs = pair_lines([Line("a")], ref)
    assert pairs == [(0, 0), (None, 1), (None, 2)]


def test_pair_by_id() -> None:
    ref = [Line("a", "l1"), Line("b", "l2"), Line("c", "l3")]
    hyp = [Line("c", "l3"), Line("a", "l1"), Line("z", "l9")]
    mode, pairs = pair_lines(ref, hyp)
    assert mode == "identifiant"
    assert pairs == [(0, 1), (1, None), (2, 0), (None, 2)]


def test_pair_falls_back_to_order() -> None:
    # Identifiants absents d'un côté, ou dupliqués : appariement par ordre.
    with_ids = [Line("a", "l1"), Line("b", "l2")]
    assert pair_lines(with_ids, [Line("a"), Line("b")])[0] == "ordre"
    duplicated = [Line("a", "l1"), Line("b", "l1")]
    assert pair_lines(duplicated, duplicated)[0] == "ordre"
    assert pair_lines(with_ids, with_ids, by_order=True)[0] == "ordre"


# --- Comparaison -----------------------------------------------------------


def test_compare_lines_missing_line_counts_as_deletions() -> None:
    ref = [Line("abcd"), Line("efgh")]
    page = compare_lines(ref, [Line("abcd")])
    assert page.count(LineStatus.MATCHED) == 1
    assert page.count(LineStatus.MISSING) == 1
    assert page.cer.errors == 4
    assert page.cer.rate == pytest.approx(4 / 8)
    assert page.lines[1].comparison.cer.counts.deletions == 4


def test_compare_lines_extra_line_counts_as_insertions() -> None:
    page = compare_lines([Line("abcd")], [Line("abcd"), Line("xy")])
    assert page.count(LineStatus.EXTRA) == 1
    assert page.cer.errors == 2
    assert page.cer.rate == pytest.approx(2 / 4)
    assert page.lines[1].comparison.cer.rate is None


def test_compare_empty_pages() -> None:
    page = compare_lines([], [])
    assert page.lines == []
    assert page.cer.rate == 0.0
    assert compare_lines([], [Line("x")]).cer.rate is None


def test_compare_nfd_hypothesis_against_nfc_reference() -> None:
    ref = [Line("Hélène Imaginaire")]
    hyp = [Line(unicodedata.normalize("NFD", "Hélène Imaginaire"))]
    assert compare_lines(ref, hyp).cer.errors == 0


def test_compare_with_normalization() -> None:
    ref, hyp = [Line("Né à Fictiville.")], [Line("NE A FICTIVILLE")]
    assert compare_lines(ref, hyp).cer.errors > 0
    options = Normalization(casefold=True, diacritics=True, punctuation=True)
    assert compare_lines(ref, hyp, options).cer.errors == 0


def test_compare_text_against_page_xml(tmp_path: Path) -> None:
    ref = tmp_path / "vue_0002.txt"
    hyp = tmp_path / "vue_0002.xml"
    ref.write_text("Jean Fictivard\nné à Fictiville\n", encoding="utf-8")
    hyp.write_text(page_xml([("l1", "Jean Fictivart"), ("l2", "né à Fictiville")]), "utf-8")
    page = compare_files(ref, hyp)
    assert page.name == "vue_0002"
    assert page.pairing == "ordre"
    assert page.cer.errors == 1
    data = page.to_dict()
    assert data["lignes_appariees"] == 2
    assert data["lignes"][0]["hypothese"] == "Jean Fictivart"


def test_compare_page_xml_by_id(tmp_path: Path) -> None:
    ref = tmp_path / "ref.xml"
    hyp = tmp_path / "hyp.xml"
    ref.write_text(page_xml([("l1", "Jean"), ("l2", "Fictivard")]), "utf-8")
    hyp.write_text(page_xml([("l2", "Fictivard"), ("l1", "Jean")]), "utf-8")
    page = compare_files(ref, hyp)
    assert page.pairing == "identifiant"
    assert page.cer.errors == 0


def test_compare_directories(tmp_path: Path) -> None:
    ref_dir, hyp_dir = tmp_path / "ref", tmp_path / "hyp"
    ref_dir.mkdir()
    hyp_dir.mkdir()
    (ref_dir / "p1.txt").write_text("abcd\n", "utf-8")
    (ref_dir / "p2.txt").write_text("efgh\n", "utf-8")  # sans hypothèse
    (ref_dir / "notes.md").write_text("ignoré", "utf-8")
    (hyp_dir / "p1.xml").write_text(page_xml([("l1", "abcX")]), "utf-8")
    (hyp_dir / "p3.txt").write_text("en trop\n", "utf-8")
    result = compare_paths(ref_dir, hyp_dir)
    assert [p.name for p in result.pages] == ["p1", "p2"]
    assert result.missing_pages == ["p2"]
    assert result.extra_pages == ["p3"]
    assert result.cer.errors == 1 + 4
    assert result.cer.rate == pytest.approx(5 / 8)
    data = result.to_dict()
    assert data["normalisation"] == ["nfc"]
    assert data["lignes_manquantes"] == 1
    assert data["confusions"][0]["nombre"] >= 1


def test_compare_paths_errors(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a", "utf-8")
    (tmp_path / "d").mkdir()
    with pytest.raises(TranscriptionError):
        compare_paths(tmp_path / "f.txt", tmp_path / "d")
    dup = tmp_path / "dup"
    dup.mkdir()
    (dup / "p.txt").write_text("a", "utf-8")
    (dup / "p.xml").write_text(page_xml([]), "utf-8")
    with pytest.raises(TranscriptionError):
        compare_paths(dup, tmp_path / "d")
