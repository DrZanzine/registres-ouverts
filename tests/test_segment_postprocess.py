# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests du post-traitement de segmentation.

DONNÉES FICTIVES : toutes les géométries ci-dessous sont synthétiques, construites pour les tests
(pages de registre imaginaires) ; elles ne proviennent d'aucune image réelle et ne contiennent aucun
texte.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from registres_ouverts.segment.__main__ import main as cli_main
from registres_ouverts.segment.postprocess import (
    PostprocessConfig,
    clip_polygon_x,
    clip_polyline_x,
    polygon_area,
    postprocess,
    postprocess_lines,
    stroke_features,
    stroke_verdict,
)

H = 40  # hauteur fictive d'une ligne
S = 45  # interligne fictif
SIZE = (2800, 1900)

_counter = [0]


def fake_line(x0: float, x1: float, y: float, height: float = H, name: str = "") -> dict[str, Any]:
    """Ligne fictive horizontale : ligne de base en ``y``, polygone de ``height`` au-dessus."""
    _counter[0] += 1
    lid = f"fictif-{name or _counter[0]}"
    top = y - 0.8 * height
    bottom = y + 0.2 * height
    return {
        "id": lid,
        "baseline": [[x0, y], [x1, y]],
        "boundary": [[x0, y], [x0, top], [x1, top], [x1, bottom], [x0, bottom], [x0, y]],
        "regions": [],
        "tags": {},
    }


def fake_act(
    prefix: str,
    y: float,
    body_x0: float,
    body_x1: float,
    margin_x0: float,
    n_lines: int = 10,
) -> tuple[list[dict[str, Any]], float]:
    """Acte fictif : numéro et nom en marge, première ligne indentée, lignes pleines, dernière
    ligne courte, deux signatures. Renvoie (lignes, ordonnée après l'acte)."""
    lines = [
        fake_line(margin_x0, margin_x0 + 90, y, name=f"{prefix}-marge-numero"),
        fake_line(margin_x0, body_x0 - 60, y + 2 * S, name=f"{prefix}-marge-nom"),
        fake_line(body_x0 + 150, body_x1, y, name=f"{prefix}-corps-00"),
    ]
    for i in range(1, n_lines):
        lines.append(
            fake_line(body_x0, body_x1 - (i % 3) * 10, y + i * S, name=f"{prefix}-corps-{i:02d}")
        )
    last = y + n_lines * S
    lines.append(fake_line(body_x0, body_x0 + 300, last, name=f"{prefix}-corps-{n_lines:02d}"))
    sig_y = last + 2 * S
    lines.append(fake_line(body_x0 + 20, body_x0 + 250, sig_y, height=55, name=f"{prefix}-sig-a"))
    lines.append(
        fake_line(body_x0 + 450, body_x0 + 700, sig_y + 10, height=55, name=f"{prefix}-sig-b")
    )
    return lines, sig_y + 3 * S


def fake_double_page() -> list[dict[str, Any]]:
    """Double page fictive : deux actes sur la page gauche, deux sur la droite."""
    lines: list[dict[str, Any]] = []
    y = 100.0
    for k in (1, 2):
        act, y = fake_act(f"g{k}", y, body_x0=450, body_x1=1380, margin_x0=150)
        lines += act
    y = 100.0
    for k in (1, 2):
        act, y = fake_act(f"d{k}", y, body_x0=1700, body_x1=2650, margin_x0=1460)
        lines += act
    return lines


def by_id(lines: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {ln["id"]: ln for ln in lines}


# --- géométrie --------------------------------------------------------------------------------


def test_clip_polygon_keeps_area() -> None:
    square = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    left = clip_polygon_x(square, 4, keep_right=False)
    right = clip_polygon_x(square, 4, keep_right=True)
    assert polygon_area(left[:-1]) == pytest.approx(40)
    assert polygon_area(right[:-1]) == pytest.approx(60)


def test_clip_polyline_interpolates() -> None:
    assert clip_polyline_x([[0, 0], [10, 10]], 5, keep_right=True) == [[5, 5], [10, 10]]
    assert clip_polyline_x([[0, 0], [10, 10]], 5, keep_right=False) == [[0, 0], [5, 5]]


# --- chaîne complète ------------------------------------------------------------------------------


def test_reading_order_zones_and_acts() -> None:
    """Page gauche puis droite ; par acte : marge, corps, signatures."""
    raw = fake_double_page()
    raw.reverse()  # l'ordre d'entrée ne doit pas compter
    result = postprocess_lines(raw, SIZE)
    lines = result.lines
    assert [ln["ordre"] for ln in lines] == list(range(1, len(lines) + 1))
    assert not result.rejected
    ids = [ln["id"].removeprefix("fictif-") for ln in lines]
    acts = [ln["acte"] for ln in lines]
    # Quatre actes numérotés dans l'ordre de lecture.
    assert sorted(set(acts)) == [1, 2, 3, 4]
    assert acts == sorted(acts)
    for k, prefix in enumerate(("g1", "g2", "d1", "d2"), start=1):
        sub = [i for i, a in zip(ids, acts, strict=True) if a == k]
        assert all(i.startswith(prefix) for i in sub)
        assert sub[:2] == [f"{prefix}-marge-numero", f"{prefix}-marge-nom"]
        assert sub[2:13] == [f"{prefix}-corps-{i:02d}" for i in range(11)]
        assert sub[13:] == [f"{prefix}-sig-a", f"{prefix}-sig-b"]
    zones = {ln["id"]: ln["zone"] for ln in lines}
    assert zones["fictif-g1-marge-nom"] == "marge"
    assert zones["fictif-g1-corps-10"] == "corps"
    assert zones["fictif-d2-sig-b"] == "signature"
    pages = {ln["id"]: ln["page"] for ln in lines}
    assert pages["fictif-g1-corps-00"] == "gauche"
    assert pages["fictif-d1-marge-numero"] == "droite"


def test_bbox_and_contract_fields() -> None:
    out = postprocess({"image": "fictif.jpg", "taille": list(SIZE), "lignes": fake_double_page()})
    assert out["image"] == "fictif.jpg"
    assert {"lignes", "rejetees", "pages"} <= out.keys()
    for ln in out["lignes"]:
        assert {"id", "baseline", "boundary", "ordre", "zone", "acte", "bbox"} <= ln.keys()
        x0, y0, x1, y1 = ln["bbox"]
        assert x0 < x1 and y0 < y1


def test_fragments_are_rejected() -> None:
    raw = fake_double_page()
    raw.append(fake_line(430, 438, 100, height=10, name="miette"))
    result = postprocess_lines(raw, SIZE)
    assert "fictif-miette" not in by_id(result.lines)
    assert [(r["id"], r["raison"]) for r in result.rejected] == [("fictif-miette", "fragment")]


def test_overflowing_body_line_is_split() -> None:
    """Une ligne du corps qui commence dans la marge est rognée ; le bout devient de la marge."""
    raw = fake_double_page()
    target = by_id(raw)["fictif-g1-corps-05"]
    x1 = target["baseline"][1][0]
    y = target["baseline"][0][1]
    raw[raw.index(target)] = fake_line(160, x1, y, name="g1-corps-05")
    result = postprocess_lines(raw, SIZE)
    lines = by_id(result.lines)
    body = lines["fictif-g1-corps-05"]
    assert body["zone"] == "corps"
    assert body["bbox"][0] > 400
    piece = lines["fictif-g1-corps-05_g"]
    assert piece["zone"] == "marge"
    assert piece["acte"] == body["acte"] == 1
    assert piece["origine"] == "fictif-g1-corps-05"
    assert piece["bbox"][2] <= body["bbox"][0]


def test_empty_page() -> None:
    """Page gauche vide : seules les lignes de la page droite sont gardées, en actes 1 et 2."""
    raw = [ln for ln in fake_double_page() if not ln["id"].startswith("fictif-g")]
    raw.append(fake_line(500, 700, 900, name="tache"))  # une seule « ligne » sur la page gauche
    out = postprocess({"image": "fictif.jpg", "taille": list(SIZE), "lignes": raw})
    assert [p["vide"] for p in out["pages"]] == [True, False]
    assert [(r["id"], r["raison"]) for r in out["rejetees"]] == [("fictif-tache", "page_vide")]
    assert {ln["acte"] for ln in out["lignes"]} == {1, 2}


def test_header_fragments_grouped_by_band() -> None:
    """En-tête imprimé émietté au-dessus du premier acte : lu par bandes, de gauche à droite."""
    acts, _ = fake_act("d1", 500, body_x0=1700, body_x1=2650, margin_x0=1460)
    header = [
        # Fragments donnés dans le désordre ; bande 1 puis bande 2.
        fake_line(2350, 2650, 160, name="entete-1c"),
        fake_line(1720, 2000, 100, name="entete-0a"),
        fake_line(2050, 2300, 158, name="entete-1b"),
        fake_line(2400, 2600, 102, name="entete-0b"),
        fake_line(1720, 1990, 163, name="entete-1a"),
        fake_line(1720, 2640, 220, name="entete-2"),
    ]
    raw = acts + header
    result = postprocess_lines(raw, SIZE)
    lines = result.lines
    head = [ln["id"].removeprefix("fictif-") for ln in lines if ln["zone"] == "entete"]
    assert head == ["entete-0a", "entete-0b", "entete-1a", "entete-1b", "entete-1c", "entete-2"]
    assert lines[0]["zone"] == "entete"
    assert all(ln["acte"] is None for ln in lines if ln["zone"] == "entete")
    assert {ln["acte"] for ln in lines if ln["zone"] != "entete"} == {1}


def test_continued_act_at_top_of_page() -> None:
    """Fin d'un acte commencé sur la vue précédente : acte 1, sans ligne de début."""
    lines = [fake_line(1700, 2650, 100 + i * S, name=f"suite-{i}") for i in range(4)]
    lines.append(fake_line(1700, 1950, 100 + 4 * S, name="suite-fin"))
    lines.append(fake_line(1750, 2000, 100 + 6 * S, height=55, name="suite-sig"))
    act, _ = fake_act("d1", 600, body_x0=1700, body_x1=2650, margin_x0=1460)
    result = postprocess_lines(lines + act, SIZE)
    zones = {ln["id"]: (ln["zone"], ln["acte"]) for ln in result.lines}
    assert zones["fictif-suite-0"] == ("corps", 1)
    assert zones["fictif-suite-fin"] == ("corps", 1)
    assert zones["fictif-suite-sig"] == ("signature", 1)
    assert zones["fictif-d1-corps-00"] == ("corps", 2)


def test_thresholds_scale_with_image() -> None:
    """Le résultat ne dépend pas de l'échelle de l'image (seuils relatifs)."""

    def scaled(raw: list[dict[str, Any]], k: float) -> list[dict[str, Any]]:
        out = json.loads(json.dumps(raw))
        for ln in out:
            for key in ("baseline", "boundary"):
                ln[key] = [[x * k, y * k] for x, y in ln[key]]
        return out

    raw = fake_double_page()
    ref = postprocess_lines(raw, SIZE).lines
    half = postprocess_lines(scaled(raw, 0.5), (SIZE[0] / 2, SIZE[1] / 2)).lines
    assert [(ln["id"], ln["zone"], ln["acte"]) for ln in ref] == [
        (ln["id"], ln["zone"], ln["acte"]) for ln in half
    ]


def test_single_page_config() -> None:
    raw = [ln for ln in fake_double_page() if ln["id"].startswith("fictif-g1")]
    result = postprocess_lines(raw, SIZE, PostprocessConfig(double_page=False))
    assert {ln["page"] for ln in result.lines} == {"unique"}
    assert {ln["acte"] for ln in result.lines} == {1}


def test_cli(tmp_path, capsys) -> None:
    src = tmp_path / "brute.json"
    dst = tmp_path / "sortie" / "segmentation.json"
    src.write_text(
        json.dumps({"image": "fictif.jpg", "taille": list(SIZE), "lignes": fake_double_page()}),
        "utf-8",
    )
    assert cli_main(["postprocess", str(src), "-o", str(dst)]) == 0
    out = json.loads(dst.read_text("utf-8"))
    assert len(out["lignes"]) == len(fake_double_page())
    assert "4 acte(s)" in capsys.readouterr().err


# --- seconde passe : signatures, marge disputée, traits -----------------------------------------


def test_signatures_after_short_line_are_not_body() -> None:
    """Après « … lecture faite » (ligne courte), des signatures alignées à gauche et dont la
    rangée atteint le bord droit restent des signatures ; un morceau décalé vers le bas sur la
    bande de la dernière ligne aussi."""
    act, _ = fake_act("g1", 100, body_x0=450, body_x1=1380, margin_x0=150)
    last_y = 100 + 10 * S
    sig_y = last_y + 1.3 * S
    act = [ln for ln in act if "-sig-" not in ln["id"]]
    act += [
        fake_line(460, 900, sig_y, height=55, name="g1-sig-gauche"),
        fake_line(930, 1375, sig_y - 5, height=55, name="g1-sig-droite"),
        fake_line(760, 830, last_y + 0.45 * S, name="g1-sig-collee"),
    ]
    page_droite, _ = fake_act("d1", 100, body_x0=1700, body_x1=2650, margin_x0=1460)
    zones = {ln["id"]: ln["zone"] for ln in postprocess_lines(act + page_droite, SIZE).lines}
    assert zones["fictif-g1-corps-10"] == "corps"
    assert zones["fictif-g1-sig-gauche"] == "signature"
    assert zones["fictif-g1-sig-droite"] == "signature"
    assert zones["fictif-g1-sig-collee"] == "signature"


def test_contested_margin_lines() -> None:
    """Juste au-dessus du début de l'acte 2 : un « N° » collé au nom qui le suit va à l'acte 2 ;
    une signature de marge collée à la marge de l'acte 1 reste à l'acte 1."""
    raw = fake_double_page()
    start = by_id(raw)["fictif-g2-corps-00"]
    y = start["baseline"][0][1]
    nom = by_id(raw)["fictif-g2-marge-nom"]
    # Le « N° » de l'acte 2 est remonté juste au-dessus de sa première ligne, collé au nom.
    raw.remove(by_id(raw)["fictif-g2-marge-numero"])
    nom_y = y + 0.5 * S
    raw[raw.index(nom)] = fake_line(150, 390, nom_y, name="g2-marge-nom")
    raw.append(fake_line(160, 260, nom_y - H - 2, name="g2-marge-numero"))
    # Signature dans la marge de la page droite, collée à une ligne de marge de l'acte d1.
    start_d = by_id(raw)["fictif-d2-corps-00"]
    yd = start_d["baseline"][0][1]
    # Comme souvent, le « N° » de l'acte d2 est un peu sous sa première ligne.
    num = by_id(raw)["fictif-d2-marge-numero"]
    raw[raw.index(num)] = fake_line(1460, 1550, yd + 1.5 * S, name="d2-marge-numero")
    raw.append(fake_line(1460, 1640, yd - 1.6 * S, name="d1-marge-mention"))
    raw.append(fake_line(1460, 1640, yd - 0.1 * S, height=70, name="d1-marge-signature"))
    acts = {ln["id"]: ln["acte"] for ln in postprocess_lines(raw, SIZE).lines}
    assert acts["fictif-g2-marge-numero"] == 2
    assert acts["fictif-d1-marge-mention"] == 3
    assert acts["fictif-d1-marge-signature"] == 3
    assert acts["fictif-d2-marge-numero"] == 4


def test_reading_order_of_tight_small_lines() -> None:
    """Petites lignes serrées (mentions marginales) : lues de haut en bas, pas par x."""
    raw = fake_double_page()
    raw += [
        fake_line(170, 380, 330, height=18, name="g1-mention-a"),
        fake_line(150, 380, 349, height=18, name="g1-mention-b"),
        fake_line(165, 380, 368, height=18, name="g1-mention-c"),
    ]
    ids = [ln["id"] for ln in postprocess_lines(raw, SIZE).lines]
    pos = [ids.index(f"fictif-g1-mention-{k}") for k in "abc"]
    assert pos == sorted(pos)


def _canvas(width: int, height: int) -> list[list[int]]:
    return [[230] * width for _ in range(height)]


def test_stroke_features_bar_vs_letters() -> None:
    """Un trait horizontal est un trait ; des jambages verticaux répétés sont du texte."""
    box = [[0, 0], [120, 0], [120, 40], [0, 40]]
    bar = _canvas(120, 40)
    for y in (24, 25, 26):
        for x in range(10, 110):
            bar[y][x] = 20
    feats = stroke_features(bar, box, (0, 0), h=40)
    assert feats is not None and stroke_verdict(feats) == "trait"
    letters = _canvas(120, 40)
    for x0 in range(10, 110, 8):
        for y in range(8, 34):
            letters[y][x0] = letters[y][x0 + 1] = 20
        for x in range(x0, x0 + 6):
            letters[8][x] = letters[33][x] = 20
    feats = stroke_features(letters, box, (0, 0), h=40)
    assert feats is not None and stroke_verdict(feats) is None


def test_strokes_rejected_with_pixel_source() -> None:
    """Avec une source de pixels, un petit trait de marge est rejeté (raison « trait ») ; sans
    elle, il est gardé. Les identifiants des autres lignes ne changent pas."""
    raw = fake_double_page()
    raw.append(fake_line(200, 300, 600, height=30, name="paraphe"))
    stroke_rows = range(588, 592)

    def pixels(bbox: tuple[int, int, int, int]) -> list[list[int]]:
        x0, y0, x1, y1 = bbox
        out = _canvas(x1 - x0, y1 - y0)
        if (x0, x1) == (200, 300):
            for y in stroke_rows:
                if y0 <= y < y1:
                    out[y - y0] = [20] * (x1 - x0)
        else:  # texte fictif : jambages réguliers, coupés en deux (deux passages par colonne)
            mid = len(out) // 2
            for k, row in enumerate(out[2:-2], start=2):
                if abs(k - mid) > 2:
                    for x in range(0, x1 - x0, 6):
                        row[x] = 20
        return out

    with_px = postprocess_lines(raw, SIZE, pixels=pixels)
    assert ("fictif-paraphe", "trait") in [(r["id"], r["raison"]) for r in with_px.rejected]
    without = postprocess_lines(raw, SIZE)
    assert "fictif-paraphe" in by_id(without.lines)
    assert {ln["id"] for ln in with_px.lines} == {ln["id"] for ln in without.lines} - {
        "fictif-paraphe"
    }
