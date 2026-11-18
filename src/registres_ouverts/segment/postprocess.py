# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-traitement géométrique d'une segmentation en lignes (kraken/blla).

La segmentation brute de kraken sur nos vues (doubles pages de registre) a plusieurs défauts :
ordre de lecture faux, lignes du corps qui débordent sur la marge, en-tête imprimé émietté,
fragments minuscules, signatures prises pour des lignes de texte, pages vides.

Ce module corrige ce qui peut l'être par la seule géométrie, sans lire le texte :

1. suppression des fragments (trop petits par rapport à la hauteur médiane des lignes) ;
2. séparation des pages gauche/droite (pli de la double page) et repérage des pages vides ;
3. sur chaque page, repérage du bord gauche de la colonne du corps (droite x = a + b·y ajustée
   sur les débuts de lignes longues), puis classement marge / corps ;
4. découpage des lignes du corps qui débordent sur la marge (la partie gauche devient une ligne de
   marge si elle est assez grande) ;
5. repérage des actes : une ligne de début d'acte est longue et fortement indentée (« L'an mil… »
   précédé d'une grande initiale), et suivie d'une ligne alignée sur le bord du corps ; le texte de
   l'acte se poursuit tant que les lignes restent alignées et régulièrement espacées, il s'arrête
   après une ligne courte (« … lecture faite. ») ; ce qui suit, jusqu'à l'acte suivant, est pris
   pour des signatures ;
6. en-tête du registre : bloc au-dessus du premier acte de la page, reconnu à ses lignes visuelles
   faites de plusieurs fragments ; les fragments sont regroupés par bande et lus de gauche à
   droite ;
7. ordre de lecture : page gauche puis droite ; sur chaque page l'en-tête, puis pour chaque acte
   la marge, le corps et les signatures.

Tous les seuils sont relatifs à la hauteur médiane des lignes ou à l'interligne médian de la vue
(voir :class:`PostprocessConfig`), jamais en pixels absolus.

Entrée et sortie : format ``segmentation_brute.json`` / ``segmentation.json`` (voir la
documentation du module ``segment``). Python pur, sans dépendance.
"""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

Point = tuple[float, float]

ZONE_HEADER = "entete"
ZONE_MARGIN = "marge"
ZONE_BODY = "corps"
ZONE_SIGNATURE = "signature"
ZONE_OTHER = "autre"

REASON_FRAGMENT = "fragment"
REASON_EMPTY_PAGE = "page_vide"
REASON_TRIMMED = "rognure_debordement"
REASON_STROKE = "trait"


@dataclass(frozen=True)
class PostprocessConfig:
    """Paramètres du post-traitement.

    Les longueurs sont exprimées en multiples de ``h`` (hauteur médiane des lignes, mesurée comme
    surface du polygone / longueur de la ligne de base) ou de ``s`` (interligne médian du corps),
    les positions en fraction de la largeur de l'image.
    """

    # Fragments : rejet si la ligne de base est plus courte que min_baseline_h * h,
    # ou si l'épaisseur est inférieure à min_thickness_h * h.
    min_baseline_h: float = 0.6
    min_thickness_h: float = 0.3
    # Ne servent à estimer h que les lignes dont la ligne de base dépasse ce multiple de la
    # médiane des longueurs (évite que les fragments faussent l'estimation).
    h_estimate_min_len_ratio: float = 0.5

    # Double page : None = automatique (image plus large que haute).
    double_page: bool | None = None
    # Le pli est cherché dans cette plage de la largeur de l'image.
    gutter_search: tuple[float, float] = (0.3, 0.7)
    # Pas de recherche du pli, en fraction de la largeur.
    gutter_step: float = 0.0025
    # Une ligne à cheval sur le pli est coupée si ses deux parties dépassent ce multiple de h.
    gutter_split_min_h: float = 3.0
    # Page vide : moins de lignes conservées que ce nombre.
    empty_page_max_lines: int = 2

    # Colonne du corps : lignes « longues » = plus larges que cette fraction de la largeur de
    # référence de la page (9e décile des largeurs de ses lignes).
    long_line_ratio: float = 0.6
    # Nombre minimal de lignes longues pour estimer la colonne du corps.
    min_long_lines: int = 3
    # Tolérance (en h) autour du mode des débuts de lignes pour l'ajustement du bord gauche.
    body_left_cluster_h: float = 1.5
    body_left_inlier_h: float = 0.75
    # Une ligne est de marge si elle finit avant bord + margin_overlap_h * h.
    margin_overlap_h: float = 1.0
    # Une ligne du corps déborde si elle commence avant bord - overflow_h * h.
    overflow_h: float = 0.75
    # Position de la coupe : bord - cut_pad_h * h.
    cut_pad_h: float = 0.3
    # La partie rognée devient une ligne de marge si elle est plus large que min_margin_piece_h * h.
    min_margin_piece_h: float = 1.5

    # Alignement sur le bord du corps (début de ligne à moins de align_h * h du bord).
    align_left_h: float = 0.75
    align_right_h: float = 1.0
    # Ligne pleine : finit à moins de full_h * h du bord droit du corps.
    full_h: float = 3.0
    # Début d'acte : indentation d'au moins start_indent_h * h, largeur d'au moins
    # start_width_ratio de la colonne, ligne suivante alignée à moins de start_next_s * s, et pas
    # de ligne pleine juste au-dessus (à moins de next_line_s * s).
    start_indent_h: float = 1.0
    start_width_ratio: float = 0.5
    start_next_s: float = 1.6
    # Largeur au-delà de laquelle un morceau à gauche sur la même bande interdit un début d'acte.
    start_left_neighbour_h: float = 3.0
    # Suite du texte d'un acte : interligne au plus next_line_s * s (ou missing_line_s * s pour
    # une ligne pleine, au cas où une ligne manquerait à la segmentation).
    next_line_s: float = 1.35
    missing_line_s: float = 2.6
    # Fragments d'une même ligne visuelle : écart de ligne de base < same_band_s * s.
    same_band_s: float = 0.5
    # Écart horizontal maximal entre deux morceaux d'une même ligne de texte, et écart vertical
    # maximal entre leurs lignes de base (un morceau plus décalé est une signature).
    chain_gap_h: float = 1.0
    chain_band_s: float = 0.3
    # En-tête : bloc d'au moins header_min_split_bands bandes faites de plusieurs fragments.
    header_min_split_bands: int = 2
    # Rattachement d'une ligne de marge à un acte : son centre est sous
    # (haut de la première ligne de l'acte - margin_attach_h * h).
    margin_attach_h: float = 0.5
    # Lignes de marge « disputées » entre deux actes : celles dont le centre est à moins de
    # margin_contest_h * h au-dessus de la ligne de base du début de l'acte suivant. Elles vont
    # à l'acte de leur plus proche voisine de marge (au-dessus ou au-dessous).
    margin_contest_h: float = 3.0
    # Ordre de lecture : deux lignes sont sur la même bande si leurs lignes de base sont à moins
    # de order_band_ratio fois la plus petite de leurs hauteurs (petites écritures serrées).
    order_band_ratio: float = 0.5

    # Traits de plume et paraphes (seulement si l'appelant fournit les pixels, voir
    # ``postprocess_lines``) ; mesurés sur les lignes de moins de trait_max_width_h * h de large.
    # vr = nombre moyen de passages encre/papier par colonne (1 = un seul trait horizontal),
    # rows = hauteur encrée en h, hr = passages par ligne de pixels par h de largeur.
    trait_max_width_h: float = 6.5
    # Rejet : un seul trait par colonne, peu haut.
    trait_max_vr: float = 1.03
    trait_max_rows_h: float = 1.2
    # Doute (zone « autre », pas de rejet) : presque un seul trait, très peu haut, peu de lettres.
    doubt_max_vr: float = 1.08
    doubt_max_rows_h: float = 0.65
    doubt_max_hr: float = 1.4
    # Seuil d'encre : pixel plus sombre que ink_ratio fois la médiane de la ligne.
    ink_ratio: float = 0.6


DEFAULT_CONFIG = PostprocessConfig()


# --------------------------------------------------------------------------------------------
# Géométrie élémentaire
# --------------------------------------------------------------------------------------------


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """Surface d'un polygone (formule du lacet), fermé ou non."""
    n = len(points)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x0, y0 = points[i][0], points[i][1]
        x1, y1 = points[(i + 1) % n][0], points[(i + 1) % n][1]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def polyline_length(points: Sequence[Sequence[float]]) -> float:
    """Longueur d'une ligne brisée."""
    return sum(
        math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        for i in range(len(points) - 1)
    )


def bounding_box(points: Iterable[Sequence[float]]) -> list[int]:
    """Rectangle englobant ``[x0, y0, x1, y1]`` (entiers)."""
    pts = list(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [
        int(math.floor(min(xs))),
        int(math.floor(min(ys))),
        int(math.ceil(max(xs))),
        int(math.ceil(max(ys))),
    ]


def clip_polygon_x(
    points: Sequence[Sequence[float]], x: float, keep_right: bool
) -> list[list[float]]:
    """Coupe un polygone par la droite verticale d'abscisse ``x`` (Sutherland-Hodgman).

    Garde la partie droite (``keep_right``) ou gauche. Le résultat est fermé (premier point
    répété) s'il n'est pas vide, comme les polygones de kraken.
    """
    pts = [list(map(float, p[:2])) for p in points]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]

    def inside(p: list[float]) -> bool:
        return p[0] >= x if keep_right else p[0] <= x

    out: list[list[float]] = []
    n = len(pts)
    for i in range(n):
        cur, nxt = pts[i], pts[(i + 1) % n]
        if inside(cur):
            out.append(cur)
        if inside(cur) != inside(nxt):
            t = (x - cur[0]) / (nxt[0] - cur[0])
            out.append([x, cur[1] + t * (nxt[1] - cur[1])])
    if out:
        out.append(list(out[0]))
    return [[round(p[0]), round(p[1])] for p in out]


def clip_polyline_x(
    points: Sequence[Sequence[float]], x: float, keep_right: bool
) -> list[list[float]]:
    """Coupe une ligne brisée (supposée orientée de gauche à droite) à l'abscisse ``x``."""
    pts = [list(map(float, p[:2])) for p in points]
    out: list[list[float]] = []
    for i, cur in enumerate(pts):
        ok = cur[0] >= x if keep_right else cur[0] <= x
        if i > 0:
            prev = pts[i - 1]
            prev_ok = prev[0] >= x if keep_right else prev[0] <= x
            if ok != prev_ok and cur[0] != prev[0]:
                t = (x - prev[0]) / (cur[0] - prev[0])
                out.append([x, prev[1] + t * (cur[1] - prev[1])])
        if ok:
            out.append(cur)
    return [[round(p[0]), round(p[1])] for p in out]


def polygon_mask(points: Sequence[Sequence[float]], width: int, height: int) -> list[list[bool]]:
    """Masque d'un polygone (coordonnées relatives au coin du rectangle) par balayage de lignes."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    edges = list(zip(pts, pts[1:] + pts[:1], strict=True))
    mask = [[False] * width for _ in range(height)]
    for y in range(height):
        yc = y + 0.5
        xs = sorted(
            a[0] + (yc - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            for a, b in edges
            if (a[1] <= yc < b[1]) or (b[1] <= yc < a[1])
        )
        row = mask[y]
        for x_in, x_out in zip(xs[::2], xs[1::2], strict=False):
            for x in range(max(0, math.ceil(x_in - 0.5)), min(width, math.floor(x_out - 0.5) + 1)):
                row[x] = True
    return mask


@dataclass(frozen=True)
class StrokeFeatures:
    """Mesures d'encre d'une ligne (voir :class:`PostprocessConfig`, section traits)."""

    vertical_runs: float
    ink_rows_h: float
    horizontal_runs_h: float


def stroke_features(
    gray: Sequence[Sequence[int]],
    boundary: Sequence[Sequence[float]],
    origin: tuple[int, int],
    h: float,
    ink_ratio: float = 0.6,
) -> StrokeFeatures | None:
    """Mesure la « texture » de l'encre d'une ligne.

    :param gray: niveaux de gris (0 = noir) du rectangle englobant, ligne par ligne.
    :param boundary: polygone de la ligne en coordonnées page.
    :param origin: coin haut-gauche ``(x0, y0)`` du rectangle dans la page.
    :param h: hauteur médiane des lignes de la vue.
    :returns: ``None`` si le polygone ne couvre aucun pixel.

    Un trait de plume horizontal (souligné, accolade, paraphe) ne croise chaque colonne qu'une
    fois et n'occupe que quelques rangées ; l'écriture croise plusieurs fois et occupe la hauteur
    d'une ligne.
    """
    height = len(gray)
    width = len(gray[0]) if height else 0
    if not width:
        return None
    rel = [(p[0] - origin[0], p[1] - origin[1]) for p in boundary]
    mask = polygon_mask(rel, width, height)
    values = sorted(gray[y][x] for y in range(height) for x in range(width) if mask[y][x])
    if not values:
        return None
    threshold = values[len(values) // 2] * ink_ratio
    ink = [[mask[y][x] and gray[y][x] < threshold for x in range(width)] for y in range(height)]
    row_runs = ink_rows = 0
    for row in ink:
        runs = sum(1 for x in range(width) if row[x] and (x == 0 or not row[x - 1]))
        if runs:
            ink_rows += 1
            row_runs += runs
    col_runs = ink_cols = 0
    for x in range(width):
        runs = sum(1 for y in range(height) if ink[y][x] and (y == 0 or not ink[y - 1][x]))
        if runs:
            ink_cols += 1
            col_runs += runs
    return StrokeFeatures(
        vertical_runs=col_runs / max(ink_cols, 1),
        ink_rows_h=ink_rows / h,
        horizontal_runs_h=row_runs / max(ink_rows, 1) / (width / h),
    )


def stroke_verdict(features: StrokeFeatures, cfg: PostprocessConfig = DEFAULT_CONFIG) -> str | None:
    """``"trait"`` (à rejeter), ``"doute"`` (à classer « autre ») ou ``None`` (texte)."""
    if features.vertical_runs <= cfg.trait_max_vr and features.ink_rows_h <= cfg.trait_max_rows_h:
        return "trait"
    if (
        features.vertical_runs <= cfg.doubt_max_vr
        and features.ink_rows_h <= cfg.doubt_max_rows_h
        and features.horizontal_runs_h <= cfg.doubt_max_hr
    ):
        return "doute"
    return None


# --------------------------------------------------------------------------------------------
# Représentation interne d'une ligne
# --------------------------------------------------------------------------------------------


@dataclass
class Line:
    """Une ligne de la segmentation et ses mesures dérivées."""

    id: str
    baseline: list[list[float]]
    boundary: list[list[float]]
    raw: dict[str, Any]
    origin: str | None = None
    zone: str = ZONE_OTHER
    act: int | None = None
    page: int = 0

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        """Recalcule les mesures après modification de la géométrie."""
        pts = self.boundary or self.baseline
        self.bbox = bounding_box(pts)
        self.x0, self.y0, self.x1, self.y1 = self.bbox
        self.length = polyline_length(self.baseline)
        area = polygon_area(self.boundary)
        self.thickness = area / self.length if self.length > 0 else 0.0
        if len(self.baseline) >= 2:
            dx = self.baseline[-1][0] - self.baseline[0][0]
            dy = self.baseline[-1][1] - self.baseline[0][1]
            self.vertical = abs(dy) > abs(dx)
        else:
            self.vertical = False
        self.bx0 = min(p[0] for p in self.baseline) if self.baseline else self.x0
        self.bx1 = max(p[0] for p in self.baseline) if self.baseline else self.x1

    @property
    def width(self) -> float:
        """Largeur du rectangle englobant."""
        return self.x1 - self.x0

    @property
    def cy(self) -> float:
        """Ordonnée moyenne de la ligne de base."""
        return sum(p[1] for p in self.baseline) / len(self.baseline)

    def y_at(self, x: float) -> float:
        """Ordonnée de la ligne de base à l'abscisse ``x`` (prolongée par constante)."""
        pts = sorted(self.baseline, key=lambda p: p[0])
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for a, b in zip(pts, pts[1:], strict=False):
            if a[0] <= x <= b[0]:
                if b[0] == a[0]:
                    return a[1]
                t = (x - a[0]) / (b[0] - a[0])
                return a[1] + t * (b[1] - a[1])
        return pts[-1][1]


def vertical_gap(upper: Line, lower: Line) -> float:
    """Écart vertical entre les lignes de base de deux lignes, mesuré à une abscisse commune."""
    lo = max(upper.bx0, lower.bx0)
    hi = min(upper.bx1, lower.bx1)
    x = (lo + hi) / 2.0
    return lower.y_at(x) - upper.y_at(x)


def split_line_x(line: Line, x: float) -> tuple[Line | None, Line | None]:
    """Coupe une ligne à l'abscisse ``x`` ; renvoie (partie gauche, partie droite).

    La partie droite garde l'identifiant d'origine ; la partie gauche reçoit ``<id>_g``.
    Une partie vide (ligne de base ou polygone dégénérés) vaut ``None``.
    """
    pieces: list[Line | None] = []
    for keep_right in (False, True):
        bl = clip_polyline_x(line.baseline, x, keep_right)
        bd = clip_polygon_x(line.boundary, x, keep_right)
        if len(bl) < 2 or len(bd) < 4 or polyline_length(bl) <= 0:
            pieces.append(None)
            continue
        new_id = line.id if keep_right else f"{line.id}_g"
        piece = Line(new_id, bl, bd, line.raw, origin=line.id, page=line.page)
        pieces.append(piece)
    return pieces[0], pieces[1]


# --------------------------------------------------------------------------------------------
# Étapes
# --------------------------------------------------------------------------------------------


def estimate_line_height(lines: Sequence[Line], cfg: PostprocessConfig = DEFAULT_CONFIG) -> float:
    """Hauteur médiane des lignes (épaisseur = surface / longueur de la ligne de base)."""
    usable = [ln for ln in lines if ln.length > 0 and ln.thickness > 0]
    if not usable:
        return 1.0
    med_len = statistics.median(ln.length for ln in usable)
    long_ones = [ln for ln in usable if ln.length >= cfg.h_estimate_min_len_ratio * med_len]
    return float(statistics.median(ln.thickness for ln in (long_ones or usable)))


def fragment_reason(line: Line, h: float, cfg: PostprocessConfig = DEFAULT_CONFIG) -> str | None:
    """Raison de rejet d'une ligne trop petite pour être du texte, ou ``None``."""
    if line.length < cfg.min_baseline_h * h:
        return REASON_FRAGMENT
    if line.thickness < cfg.min_thickness_h * h:
        return REASON_FRAGMENT
    return None


def find_gutter(
    lines: Sequence[Line], width: float, cfg: PostprocessConfig = DEFAULT_CONFIG
) -> float:
    """Abscisse du pli d'une double page.

    Pour chaque abscisse candidate, on somme la pénétration ``min(x - x0, x1 - x)`` des lignes qui
    la traversent ; le pli est l'abscisse de coût minimal (au milieu de la plus large plage de coût
    minimal, au plus près du centre de l'image en cas d'égalité).
    """
    lo, hi = cfg.gutter_search
    step = max(1.0, cfg.gutter_step * width)
    candidates: list[float] = []
    x = lo * width
    while x <= hi * width:
        candidates.append(x)
        x += step
    if not candidates:
        return width / 2.0
    costs = []
    for c in candidates:
        cost = 0.0
        for ln in lines:
            if ln.x0 < c < ln.x1:
                cost += min(c - ln.x0, ln.x1 - c)
        costs.append(cost)
    best = min(costs)
    # Plages contiguës de coût minimal ; on garde la plus proche du centre, prise en son milieu.
    runs: list[tuple[int, int]] = []
    start = None
    for i, cost in enumerate(costs):
        if cost <= best + 1e-9:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(costs) - 1))

    # La plage la plus large l'emporte (une page vide laisse un grand blanc), puis la plus
    # proche du centre.
    run = max(
        runs,
        key=lambda r: (r[1] - r[0], -abs((candidates[r[0]] + candidates[r[1]]) / 2 - width / 2)),
    )
    lo_x, hi_x = candidates[run[0]], candidates[run[1]]
    # Page vide d'un côté : le pli est au bord du texte de l'autre page, pas au milieu du blanc.
    if not any(ln.x1 <= lo_x for ln in lines):
        return hi_x
    if not any(ln.x0 >= hi_x for ln in lines):
        return lo_x
    return (lo_x + hi_x) / 2.0


@dataclass
class BodyColumn:
    """Colonne du corps d'une page : bord gauche ``left + slope * y``, bord droit ``right``."""

    left: float
    slope: float
    right: float

    def left_at(self, y: float) -> float:
        """Bord gauche du corps à l'ordonnée ``y``."""
        return self.left + self.slope * y

    @property
    def width(self) -> float:
        """Largeur approximative de la colonne (au milieu de la page)."""
        return self.right - self.left


def _fit_line(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Moindres carrés x = a + b·y ; renvoie (a, b)."""
    n = len(points)
    my = sum(p[1] for p in points) / n
    mx = sum(p[0] for p in points) / n
    var = sum((p[1] - my) ** 2 for p in points)
    if n < 3 or var == 0:
        return mx, 0.0
    b = sum((p[1] - my) * (p[0] - mx) for p in points) / var
    return mx - b * my, b


def find_body_column(
    lines: Sequence[Line],
    page_x0: float,
    page_x1: float,
    h: float,
    cfg: PostprocessConfig = DEFAULT_CONFIG,
) -> BodyColumn | None:
    """Estime la colonne du corps d'une page à partir des débuts des lignes longues.

    Le bord gauche est le mode (par classes de largeur ``h``) des débuts de lignes longues,
    affiné par une droite ``x = a + b·y`` (la page peut être légèrement inclinée ou bombée).
    """
    flat = [ln for ln in lines if not ln.vertical]
    if not flat:
        return None
    widths = sorted(ln.width for ln in flat)
    ref = widths[int(0.9 * (len(widths) - 1))]
    longs = [ln for ln in flat if ln.width >= cfg.long_line_ratio * ref]
    if len(longs) < cfg.min_long_lines:
        return None
    bins: dict[int, int] = {}
    for ln in longs:
        k = int((ln.x0 - page_x0) // h)
        bins[k] = bins.get(k, 0) + 1
    # Mode lissé sur trois classes, pour ne pas couper un amas en deux.
    best_k = max(bins, key=lambda k: (bins.get(k - 1, 0) + bins[k] + bins.get(k + 1, 0), -k))
    mode = page_x0 + (best_k + 0.5) * h
    cluster = [ln for ln in longs if abs(ln.x0 - mode) <= cfg.body_left_cluster_h * h]
    pts = [(float(ln.x0), (ln.y0 + ln.y1) / 2.0) for ln in cluster]
    a, b = _fit_line(pts)
    for _ in range(2):
        inl = [p for p in pts if abs(p[0] - (a + b * p[1])) <= cfg.body_left_inlier_h * h]
        if len(inl) < 3:
            break
        a, b = _fit_line(inl)
    # On prend le bord un peu à gauche de la droite moyenne (bas de la dispersion des débuts).
    residuals = sorted(p[0] - (a + b * p[1]) for p in pts if abs(p[0] - (a + b * p[1])) <= h)
    if residuals:
        a += residuals[max(0, len(residuals) // 10)]
    rights = sorted(ln.x1 for ln in longs)
    right = rights[int(0.75 * (len(rights) - 1))]
    return BodyColumn(left=a, slope=b, right=float(right))


def group_bands(
    lines: Sequence[Line], tol: float, thickness_ratio: float | None = None
) -> list[list[Line]]:
    """Regroupe des lignes en bandes horizontales (même ligne visuelle), triées de haut en bas,
    chaque bande triée de gauche à droite.

    Deux lignes sont sur la même bande si l'écart de leurs lignes de base est inférieur à ``tol``,
    et, si ``thickness_ratio`` est donné, à ``thickness_ratio`` fois la plus mince des deux
    (petites écritures serrées des mentions marginales).
    """
    bands: list[list[Line]] = []

    def limit(a: Line, b: Line) -> float:
        if thickness_ratio is None:
            return tol
        return min(tol, thickness_ratio * min(a.thickness, b.thickness))

    for ln in sorted(lines, key=lambda ln: ln.cy):
        placed = False
        for band in reversed(bands[-3:]):
            ref = min(band, key=lambda b: abs(vertical_gap(b, ln)))
            if abs(vertical_gap(ref, ln)) < limit(ref, ln):
                band.append(ln)
                placed = True
                break
        if not placed:
            bands.append([ln])
    for band in bands:
        band.sort(key=lambda ln: ln.x0)
    bands.sort(key=lambda b: min(ln.cy for ln in b))
    return bands


# --------------------------------------------------------------------------------------------
# Analyse d'une page
# --------------------------------------------------------------------------------------------


@dataclass
class Act:
    """Un acte repéré sur une page."""

    top: float
    start: Line | None
    body: list[Line] = field(default_factory=list)
    margin: list[Line] = field(default_factory=list)
    signatures: list[Line] = field(default_factory=list)
    other: list[Line] = field(default_factory=list)
    end: Line | None = None
    continued: bool = False


@dataclass
class PageLayout:
    """Résultat de l'analyse d'une page."""

    index: int
    x0: float
    x1: float
    column: BodyColumn | None
    header: list[Line] = field(default_factory=list)
    acts: list[Act] = field(default_factory=list)
    other: list[Line] = field(default_factory=list)
    trimmed: list[Line] = field(default_factory=list)


class _PageAnalyzer:
    """Analyse d'une page : marge/corps, actes, signatures, en-tête."""

    def __init__(
        self,
        lines: list[Line],
        index: int,
        x0: float,
        x1: float,
        h: float,
        s: float | None,
        cfg: PostprocessConfig,
    ) -> None:
        self.cfg = cfg
        self.h = h
        self.s_hint = s
        self.layout = PageLayout(index=index, x0=x0, x1=x1, column=None)
        self.s = s or h
        self.lines = lines

    # -- colonne et débordements ---------------------------------------------------------------

    def split_margin_body(self) -> tuple[list[Line], list[Line], list[Line]]:
        """Sépare marge, corps et lignes tournées ; rogne les débordements sur la marge."""
        cfg, h = self.cfg, self.h
        col = find_body_column(self.lines, self.layout.x0, self.layout.x1, h, cfg)
        self.layout.column = col
        margin: list[Line] = []
        body: list[Line] = []
        rotated: list[Line] = []
        for ln in self.lines:
            if ln.vertical:
                rotated.append(ln)
                continue
            if col is None:
                body.append(ln)
                continue
            left = col.left_at((ln.y0 + ln.y1) / 2.0)
            if ln.x1 <= left + cfg.margin_overlap_h * h:
                margin.append(ln)
            elif ln.x0 < left - cfg.overflow_h * h:
                cut = left - cfg.cut_pad_h * h
                lpart, rpart = split_line_x(ln, cut)
                if rpart is None:
                    margin.append(ln)
                    continue
                body.append(rpart)
                if lpart is not None:
                    if lpart.width >= cfg.min_margin_piece_h * h and lpart.thickness > 0:
                        margin.append(lpart)
                    else:
                        self.layout.trimmed.append(lpart)
            else:
                body.append(ln)
        return margin, body, rotated

    # -- prédicats -----------------------------------------------------------------------------

    def aligned(self, ln: Line) -> bool:
        col = self.layout.column
        assert col is not None
        left = col.left_at((ln.y0 + ln.y1) / 2.0)
        return -self.cfg.align_left_h * self.h <= ln.x0 - left <= self.cfg.align_right_h * self.h

    def full(self, ln: Line) -> bool:
        col = self.layout.column
        assert col is not None
        return ln.x1 >= col.right - self.cfg.full_h * self.h

    def indent(self, ln: Line) -> float:
        col = self.layout.column
        assert col is not None
        return ln.x0 - col.left_at((ln.y0 + ln.y1) / 2.0)

    # -- interligne ----------------------------------------------------------------------------

    def estimate_spacing(self, body: Sequence[Line]) -> float:
        """Interligne médian des lignes pleines et alignées consécutives."""
        good = sorted((ln for ln in body if self.full(ln)), key=lambda ln: ln.cy)
        gaps = []
        for a, b in zip(good, good[1:], strict=False):
            g = vertical_gap(a, b)
            if 0.5 * self.h < g < 2.0 * self.h:
                gaps.append(g)
        if len(gaps) >= 3:
            return float(statistics.median(gaps))
        return self.s_hint or self.h

    # -- analyse -------------------------------------------------------------------------------

    def next_text_line(
        self, current: Line, candidates: Sequence[Line], s: float, used: set[int]
    ) -> Line | None:
        """Ligne de texte suivante d'un acte, ou ``None`` si le texte s'arrête."""
        cfg = self.cfg
        below = [
            c
            for c in candidates
            if id(c) not in used and vertical_gap(current, c) > cfg.same_band_s * s
        ]
        if not below:
            return None
        nxt = min(below, key=lambda c: vertical_gap(current, c))
        gap = vertical_gap(current, nxt)
        # Plusieurs fragments sur la bande suivante : on prend le plus à gauche.
        band = [c for c in below if abs(vertical_gap(nxt, c)) < cfg.same_band_s * s]
        nxt = min(band, key=lambda c: c.x0)
        if not self.aligned(nxt):
            return None
        cur_full = self.full(current)
        nxt_full = self.full(nxt) or any(self.full(c) for c in self.chained_right(nxt, band))
        if gap <= cfg.next_line_s * s and cur_full:
            return nxt
        # Après une ligne courte (« … lecture faite »), seule une ligne pleine par elle-même
        # continue le texte : des signatures voisines ne doivent pas la rendre « pleine ».
        if gap <= cfg.next_line_s * s and self.full(nxt):
            return nxt
        if gap <= cfg.missing_line_s * s and nxt_full and cur_full:
            return nxt
        return None

    def is_start(self, ln: Line, body: Sequence[Line], s: float) -> bool:
        """Vrai si ``ln`` ressemble à la première ligne d'un acte."""
        cfg, col = self.cfg, self.layout.column
        assert col is not None
        if self.indent(ln) < cfg.start_indent_h * self.h:
            return False
        if ln.x1 - ln.x0 < cfg.start_width_ratio * col.width:
            return False
        # Un fragment précédé, sur la même bande, d'un morceau de texte n'est pas un début d'acte
        # (cas de l'en-tête imprimé émietté). Une petite initiale ornée isolée est tolérée.
        if any(
            c is not ln
            and c.x1 <= ln.x0 + self.h
            and c.width > cfg.start_left_neighbour_h * self.h
            and abs(vertical_gap(ln, c)) < cfg.same_band_s * s
            for c in body
        ):
            return False
        below = [c for c in body if c is not ln and vertical_gap(ln, c) > cfg.same_band_s * s]
        if not below:
            return False
        nxt = min(below, key=lambda c: vertical_gap(ln, c))
        band = [c for c in below if abs(vertical_gap(nxt, c)) < cfg.same_band_s * s]
        nxt = min(band, key=lambda c: c.x0)
        if not (self.aligned(nxt) and vertical_gap(ln, nxt) <= cfg.start_next_s * s):
            return False
        above = [
            c
            for c in body
            if c is not ln and cfg.same_band_s * s < vertical_gap(c, ln) <= cfg.next_line_s * s
        ]
        return not any(self.full(c) and self.aligned(c) for c in above)

    def chained_right(self, row: Line, band: Sequence[Line]) -> list[Line]:
        """Morceaux d'une bande qui prolongent ``row`` vers la droite sans grand blanc."""
        out: list[Line] = []
        edge = row.x1
        for c in sorted(band, key=lambda c: c.x0):
            if c is row or c.x0 < row.x0:
                continue
            if abs(vertical_gap(row, c)) > self.cfg.chain_band_s * self.s:
                continue
            if c.x0 - edge <= self.cfg.chain_gap_h * self.h:
                out.append(c)
                edge = max(edge, c.x1)
        return out

    def resolve_contested_margin(
        self, margin: Sequence[Line], acts: Sequence[Act], owner: dict[int, Act | None]
    ) -> None:
        """Départage les lignes de marge situées entre deux actes.

        Juste au-dessus du début d'un acte, la marge peut porter le « N° » de cet acte ou une
        signature de l'acte précédent. Chaque ligne disputée suit sa plus proche voisine de marge
        (chevauchement horizontal requis), traitée de bas en haut ; à écart égal, l'acte suivant.
        """
        cfg, h = self.cfg, self.h
        for prev, act in zip(acts, acts[1:], strict=False):
            if act.start is None:
                continue
            level = act.start.y_at(act.start.x0)
            contested = [
                ln
                for ln in margin
                if level - cfg.margin_contest_h * h <= (ln.y0 + ln.y1) / 2.0 < level
                and owner.get(id(ln)) in (prev, act)
            ]
            for ln in sorted(contested, key=lambda ln: -ln.cy):
                others = [
                    o
                    for o in margin
                    if o is not ln
                    and min(o.x1, ln.x1) > max(o.x0, ln.x0)
                    and owner.get(id(o)) in (prev, act)
                ]
                above = [o for o in others if o.cy < ln.cy]
                below = [o for o in others if o.cy > ln.cy]
                gap_above = min((ln.y0 - o.y1 for o in above), default=math.inf)
                gap_below = min((o.y0 - ln.y1 for o in below), default=math.inf)
                if not above or gap_below <= gap_above:
                    best = min(below, key=lambda o: o.y0 - ln.y1) if below else None
                else:
                    best = max(above, key=lambda o: o.y1)
                if best is not None:
                    owner[id(ln)] = owner[id(best)]

    def analyze(self) -> PageLayout:
        cfg, h = self.cfg, self.h
        margin, body, rotated = self.split_margin_body()
        layout = self.layout
        if layout.column is None:
            layout.other.extend(self.lines)
            return layout
        s = self.estimate_spacing(body)
        self.s = s
        band_tol = cfg.same_band_s * s
        body_sorted = sorted(body, key=lambda ln: ln.cy)
        starts = [ln for ln in body_sorted if self.is_start(ln, body_sorted, s)]

        # Bloc au-dessus du premier début d'acte : en-tête, suite d'un acte, ou autre.
        first_top = starts[0].y0 if starts else math.inf
        pre = [ln for ln in body_sorted if ln.cy < first_top and ln not in starts]
        acts: list[Act] = []
        assigned: set[int] = set()
        if pre:
            bands = group_bands(pre, band_tol)
            split_bands = sum(
                1
                for b in bands
                if len(b) >= 2 and any(b2.x0 - b1.x1 > h for b1, b2 in zip(b, b[1:], strict=False))
            )
            if split_bands >= cfg.header_min_split_bands:
                layout.header = [ln for b in bands for ln in b]
                assigned.update(id(ln) for ln in pre)
            else:
                head = next((ln for ln in pre if self.aligned(ln) and self.full(ln)), None)
                if head is not None:
                    acts.append(
                        Act(top=float(min(ln.y0 for ln in pre)), start=head, continued=True)
                    )
        for st in starts:
            acts.append(Act(top=st.y0, start=st))
        acts.sort(key=lambda a: a.top)

        # Texte de chaque acte.
        for i, act in enumerate(acts):
            limit = acts[i + 1].start.y0 if i + 1 < len(acts) else math.inf  # type: ignore[union-attr]
            pool = [ln for ln in body_sorted if id(ln) not in assigned and ln.cy < limit]
            cur = act.start
            assert cur is not None
            used = {id(cur)}
            chain = [cur]
            while True:
                nxt = self.next_text_line(cur, pool, s, used)
                if nxt is None:
                    break
                chain.append(nxt)
                used.add(id(nxt))
                cur = nxt
            act.end = cur
            end_y = cur.cy
            end_band = [ln for ln in pool if abs(vertical_gap(cur, ln)) < band_tol]
            chained = self.chained_right(cur, end_band)
            for ln in pool:
                if id(ln) in assigned or ln.cy < act.start.cy - band_tol:
                    continue
                gap = vertical_gap(cur, ln)
                if ln in chain or gap <= -band_tol:
                    in_body = True
                elif abs(gap) < band_tol:
                    in_body = ln in chained
                else:
                    in_body = False
                if in_body:
                    act.body.append(ln)
                    assigned.add(id(ln))
            for ln in pool:
                if id(ln) not in assigned and ln.cy > end_y - band_tol:
                    act.signatures.append(ln)
                    assigned.add(id(ln))
        # Reste du corps : avant le premier acte (hors en-tête) ou non rattaché.
        for ln in body_sorted:
            if id(ln) not in assigned:
                layout.other.append(ln)

        # Marge et lignes tournées : rattachées à l'acte dont le début est au-dessus.
        owner: dict[int, Act | None] = {}
        for ln in margin + rotated:
            centre = (ln.y0 + ln.y1) / 2.0
            target = None
            for act in acts:
                if centre >= act.top - cfg.margin_attach_h * h:
                    target = act
            owner[id(ln)] = target
        self.resolve_contested_margin(margin, acts, owner)
        for ln in sorted(margin + rotated, key=lambda ln: ln.cy):
            target = owner[id(ln)]
            if target is None:
                layout.other.append(ln)
            elif ln.vertical:
                target.other.append(ln)
            else:
                target.margin.append(ln)
        layout.acts = acts
        return layout


# --------------------------------------------------------------------------------------------
# Point d'entrée
# --------------------------------------------------------------------------------------------


def _ordered(lines: Sequence[Line], tol: float, ratio: float) -> list[Line]:
    return [ln for band in group_bands(lines, tol, ratio) for ln in band]


@dataclass
class PostprocessResult:
    """Résultat du post-traitement : lignes conservées, rejetées, et description des pages."""

    lines: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    pages: list[dict[str, Any]]


def split_pages(
    lines: Sequence[Line],
    width: float,
    height: float,
    h: float,
    cfg: PostprocessConfig = DEFAULT_CONFIG,
) -> list[tuple[float, float, list[Line]]]:
    """Répartit les lignes entre les pages ; renvoie ``[(x0, x1, lignes), …]`` de gauche à droite.

    Une ligne à cheval sur le pli est coupée si ses deux parties sont assez grandes, sinon elle va
    entièrement du côté où elle est le plus.
    """
    double = cfg.double_page if cfg.double_page is not None else width > height
    if not double:
        return [(0.0, width, list(lines))]
    gutter = find_gutter(lines, width, cfg)
    pages: list[tuple[float, float, list[Line]]] = [(0.0, gutter, []), (gutter, width, [])]
    for ln in lines:
        left_part = max(0.0, min(ln.x1, gutter) - ln.x0)
        right_part = max(0.0, ln.x1 - max(ln.x0, gutter))
        if min(left_part, right_part) >= cfg.gutter_split_min_h * h and not ln.vertical:
            lp, rp = split_line_x(ln, gutter)
            if lp is not None:
                pages[0][2].append(lp)
            if rp is not None:
                pages[1][2].append(rp)
        else:
            pages[0 if left_part >= right_part else 1][2].append(ln)
    return pages


PixelSource = Callable[[tuple[int, int, int, int]], Sequence[Sequence[int]]]


def postprocess_lines(
    raw_lines: Sequence[dict[str, Any]],
    size: Sequence[int],
    cfg: PostprocessConfig = DEFAULT_CONFIG,
    pixels: PixelSource | None = None,
) -> PostprocessResult:
    """Post-traite une liste de lignes kraken.

    :param raw_lines: lignes au format ``segmentation_brute.json`` (``id``, ``baseline``,
        ``boundary`` ; les autres champs sont recopiés).
    :param size: taille de l'image ``[largeur, hauteur]``.
    :param pixels: facultatif ; fonction qui renvoie les niveaux de gris (0 = noir) d'un
        rectangle ``(x0, y0, x1, y1)`` de l'image, ligne par ligne. Si elle est fournie, les
        petites lignes sont examinées : un trait de plume est rejeté (raison ``trait``), une ligne
        douteuse est classée ``autre``. Sans elle, ce test est sauté.
    :returns: lignes conservées triées par ``ordre`` (avec ``ordre``, ``zone``, ``acte``,
        ``bbox``, ``page``), lignes rejetées (avec ``raison``) et pages.
    """
    width, height = float(size[0]), float(size[1])
    lines = [
        Line(str(r["id"]), [list(p) for p in r["baseline"]], [list(p) for p in r["boundary"]], r)
        for r in raw_lines
        if r.get("baseline") and r.get("boundary")
    ]
    rejected: list[tuple[Line, str]] = []
    h = estimate_line_height(lines, cfg)

    kept: list[Line] = []
    for ln in lines:
        reason = fragment_reason(ln, h, cfg)
        if reason:
            rejected.append((ln, reason))
        else:
            kept.append(ln)

    doubtful: set[int] = set()
    if pixels is not None:
        still: list[Line] = []
        for ln in kept:
            verdict = None
            if not ln.vertical and ln.width <= cfg.trait_max_width_h * h:
                feats = stroke_features(
                    pixels((ln.x0, ln.y0, ln.x1, ln.y1)),
                    ln.boundary,
                    (ln.x0, ln.y0),
                    h,
                    cfg.ink_ratio,
                )
                verdict = stroke_verdict(feats, cfg) if feats else None
            if verdict == "trait":
                rejected.append((ln, REASON_STROKE))
                continue
            if verdict == "doute":
                doubtful.add(id(ln))
            still.append(ln)
        kept = still

    pages = split_pages(kept, width, height, h, cfg)
    names = ["gauche", "droite"] if len(pages) == 2 else ["unique"]
    page_info: list[dict[str, Any]] = []
    layouts: list[PageLayout] = []
    for index, (px0, px1, plines) in enumerate(pages):
        info: dict[str, Any] = {
            "page": names[index],
            "bbox": [round(px0), 0, round(px1), round(height)],
            "vide": len(plines) <= cfg.empty_page_max_lines,
        }
        page_info.append(info)
        if info["vide"]:
            rejected.extend((ln, REASON_EMPTY_PAGE) for ln in plines)
            continue
        for ln in plines:
            ln.page = index
        layout = _PageAnalyzer(plines, index, px0, px1, h, None, cfg).analyze()
        if layout.column is not None:
            col = layout.column
            info["corps_x"] = [round(col.left_at(0)), round(col.left_at(height)), round(col.right)]
        info["entete"] = bool(layout.header)
        rejected.extend((ln, REASON_TRIMMED) for ln in layout.trimmed)
        layouts.append(layout)

    # Ordre de lecture et numérotation des actes.
    ordered: list[Line] = []
    act_no = 0
    tol = cfg.same_band_s * h
    for layout in layouts:
        for ln in _ordered(layout.header, tol, cfg.order_band_ratio):
            ln.zone, ln.act = ZONE_HEADER, None
            ordered.append(ln)
        for act in layout.acts:
            act_no += 1
            for group, zone in (
                (act.margin, ZONE_MARGIN),
                (act.other, ZONE_OTHER),
                (act.body, ZONE_BODY),
                (act.signatures, ZONE_SIGNATURE),
            ):
                for ln in _ordered(group, tol, cfg.order_band_ratio):
                    ln.zone, ln.act = (ZONE_OTHER if id(ln) in doubtful else zone), act_no
                    ordered.append(ln)
        for ln in _ordered(layout.other, tol, cfg.order_band_ratio):
            ln.zone, ln.act = ZONE_OTHER, None
            ordered.append(ln)

    out = [_export(ln, rank, names) for rank, ln in enumerate(ordered, start=1)]
    rej = []
    for ln, reason in rejected:
        item: dict[str, Any] = {
            "id": ln.id,
            "raison": reason,
            "bbox": ln.bbox,
            "baseline": _int_points(ln.baseline),
            "boundary": _int_points(ln.boundary),
        }
        if ln.origin and ln.origin != ln.id:
            item["origine"] = ln.origin
        rej.append(item)
    return PostprocessResult(lines=out, rejected=rej, pages=page_info)


def _int_points(points: Sequence[Sequence[float]]) -> list[list[int]]:
    return [[int(round(p[0])), int(round(p[1]))] for p in points]


def _export(ln: Line, rank: int, page_names: Sequence[str]) -> dict[str, Any]:
    item = copy.deepcopy({k: v for k, v in ln.raw.items() if k not in ("baseline", "boundary")})
    item["id"] = ln.id
    item["baseline"] = _int_points(ln.baseline)
    item["boundary"] = _int_points(ln.boundary)
    if ln.origin and ln.origin != ln.id:
        item["origine"] = ln.origin
    item["ordre"] = rank
    item["zone"] = ln.zone
    item["acte"] = ln.act
    item["bbox"] = ln.bbox
    item["page"] = page_names[ln.page]
    return item


def postprocess(
    data: dict[str, Any],
    cfg: PostprocessConfig = DEFAULT_CONFIG,
    pixels: PixelSource | None = None,
) -> dict[str, Any]:
    """Post-traite un document ``segmentation_brute.json`` et renvoie ``segmentation.json``.

    Les clés du document d'entrée sont recopiées ; ``lignes`` est remplacé par les lignes
    conservées, et ``rejetees`` et ``pages`` sont ajoutés. ``pixels`` : voir
    :func:`postprocess_lines`.
    """
    result = postprocess_lines(data.get("lignes", []), data["taille"], cfg, pixels)
    out = {k: copy.deepcopy(v) for k, v in data.items() if k != "lignes"}
    out["lignes"] = result.lines
    out["rejetees"] = result.rejected
    out["pages"] = result.pages
    return out
