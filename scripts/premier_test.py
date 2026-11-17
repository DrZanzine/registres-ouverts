# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Premier test brut : segmentation et transcription kraken, sans entraînement.

Sur quelques images de data/raw/, le script :

1. segmente la page en lignes (modèle de segmentation kraken, par défaut
   celui fourni avec kraken) ;
2. transcrit chaque ligne avec chacun des modèles donnés (--rec-model,
   répétable), la segmentation étant faite une seule fois ;
3. écrit dans outputs/ (non versionné) :
   - ``<page>.lignes.jpg`` : l'image avec les lignes détectées dessinées ;
   - ``<modèle>/<page>.page.xml`` : PAGE XML avec le texte ;
   - ``<modèle>/<page>.txt`` : texte brut, une ligne par ligne détectée.

Aucune évaluation chiffrée : il s'agit seulement de voir le niveau brut.

Usage :
    uv run --extra htr python scripts/premier_test.py \\
        --rec-model models/a.mlmodel [--rec-model models/b.mlmodel] [-n 3]
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
LINE_COLOUR = (220, 30, 30)
BASELINE_COLOUR = (30, 90, 220)


def find_images(raw_dir: Path, count: int) -> list[Path]:
    """Renvoie les ``count`` premières images de ``raw_dir`` (ordre alphabétique)."""
    images = sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    return images[:count]


def draw_lines(image: Image.Image, lines: list) -> Image.Image:
    """Dessine le contour, la ligne de base et le numéro de chaque ligne."""
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width = max(2, canvas.width // 800)
    for index, line in enumerate(lines, start=1):
        if line.boundary:
            draw.polygon([tuple(p) for p in line.boundary], outline=LINE_COLOUR, width=width)
        if line.baseline:
            draw.line([tuple(p) for p in line.baseline], fill=BASELINE_COLOUR, width=width)
            x, y = line.baseline[0]
            size = 12 * width
            draw.text(
                (max(0, x - 3 * size), max(0, y - size)),
                str(index),
                fill=LINE_COLOUR,
                font_size=size,
            )
    return canvas


def mean_confidence(lines: list) -> float | None:
    """Confiance moyenne par caractère sur la page (indicative, non calibrée)."""
    values = [c for line in lines for c in getattr(line, "confidences", []) or []]
    return sum(values) / len(values) if values else None


def write_page_xml(segmentation, image: Image.Image, name: str, path: Path) -> None:
    """Écrit une segmentation (avec ou sans texte) en PAGE XML."""
    from kraken.serialization import serialize

    segmentation = dataclasses.replace(segmentation, imagename=name)
    path.write_text(serialize(segmentation, image_size=image.size, template="pagexml"), "utf-8")


def process_page(
    path: Path, seg_model, rec_models: dict, seg_config, rec_config, out_dir: Path
) -> None:
    """Segmente une page une fois, puis la transcrit avec chaque modèle.

    Sorties : ``<page>.lignes.jpg`` dans ``out_dir`` ; pour chaque modèle,
    ``<modèle>/<page>.page.xml`` et ``<modèle>/<page>.txt``. Sans modèle de
    transcription, ``<page>.page.xml`` (lignes seules) dans ``out_dir``.
    """
    image = Image.open(path)
    stem = path.stem

    start = time.perf_counter()
    segmentation = seg_model.predict(image, seg_config)
    seg_seconds = time.perf_counter() - start
    print(f"  {len(segmentation.lines)} lignes détectées ({seg_seconds:.0f} s)", flush=True)
    draw_lines(image, segmentation.lines).save(out_dir / f"{stem}.lignes.jpg", quality=85)

    if not rec_models:
        write_page_xml(segmentation, image, path.name, out_dir / f"{stem}.page.xml")
        return

    for name, rec_model in rec_models.items():
        start = time.perf_counter()
        lines = list(rec_model.predict(image, segmentation, rec_config))
        confidence = mean_confidence(lines)
        shown = f", confiance moyenne {confidence:.2f}" if confidence is not None else ""
        print(f"  {name} : {time.perf_counter() - start:.0f} s{shown}", flush=True)

        model_dir = out_dir / name
        model_dir.mkdir(exist_ok=True)
        transcribed = dataclasses.replace(segmentation, lines=lines)
        write_page_xml(transcribed, image, path.name, model_dir / f"{stem}.page.xml")
        # Les modèles sont entraînés en NFD : le texte brut est recomposé en NFC
        # pour être lisible partout, sans autre modification.
        text = "\n".join(unicodedata.normalize("NFC", line.prediction) for line in lines)
        (model_dir / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="*", type=Path, help="images (défaut : les n premières)")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("-n", "--count", type=int, default=3)
    parser.add_argument("--seg-model", type=Path, default=None, help="défaut : modèle kraken")
    parser.add_argument(
        "--rec-model",
        type=Path,
        action="append",
        default=[],
        help="modèle de transcription (répétable : un dossier de sortie par modèle)",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--accelerator", default="auto", help="auto, cpu ou gpu")
    args = parser.parse_args(argv)

    try:
        from kraken.configs import RecognitionInferenceConfig, SegmentationInferenceConfig
        from kraken.tasks import RecognitionTaskModel, SegmentationTaskModel
    except ImportError:
        parser.error("kraken n'est pas installé : lancer avec `uv run --extra htr ...`")

    images = args.images or find_images(args.raw_dir, args.count)
    if not images:
        parser.error(f"aucune image trouvée dans {args.raw_dir}")
    if not args.rec_model:
        print("Pas de --rec-model : segmentation seule, sans transcription.", file=sys.stderr)

    threads = os.cpu_count() or 1
    common = {"accelerator": args.accelerator, "num_threads": threads}
    seg_config = SegmentationInferenceConfig(**common)
    rec_config = RecognitionInferenceConfig(**common)

    seg_model = SegmentationTaskModel.load_model(args.seg_model)
    rec_models = {p.stem: RecognitionTaskModel.load_model(p) for p in args.rec_model}

    args.output.mkdir(parents=True, exist_ok=True)
    for path in images:
        print(path, flush=True)
        process_page(path, seg_model, rec_models, seg_config, rec_config, args.output)
    print(f"Sorties écrites dans {args.output}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
