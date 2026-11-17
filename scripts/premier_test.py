# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Premier test brut : segmentation et transcription kraken, sans entraînement.

Sur quelques images de data/raw/, le script :

1. segmente la page en lignes (modèle de segmentation kraken, par défaut
   celui fourni avec kraken) ;
2. transcrit chaque ligne si un modèle de transcription est donné ;
3. écrit dans outputs/ (non versionné), pour chaque page :
   - ``<page>.page.xml`` : PAGE XML (lignes, et texte si transcription) ;
   - ``<page>.txt`` : texte brut, une ligne par ligne détectée ;
   - ``<page>.lignes.jpg`` : l'image avec les lignes détectées dessinées.

Aucune évaluation chiffrée : il s'agit seulement de voir le niveau brut.

Usage :
    uv run --extra htr python scripts/premier_test.py \\
        --rec-model models/<modele>.mlmodel [--seg-model ...] [-n 3]
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
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


def process_page(path: Path, seg_model, rec_model, seg_config, rec_config, out_dir: Path) -> None:
    """Segmente, transcrit et écrit les sorties d'une page."""
    from kraken.serialization import serialize

    image = Image.open(path)
    stem = path.stem

    start = time.perf_counter()
    segmentation = seg_model.predict(image, seg_config)
    seg_seconds = time.perf_counter() - start
    print(f"  {len(segmentation.lines)} lignes détectées ({seg_seconds:.0f} s)")

    lines = segmentation.lines
    if rec_model is not None:
        start = time.perf_counter()
        lines = list(rec_model.predict(image, segmentation, rec_config))
        segmentation = dataclasses.replace(segmentation, lines=lines)
        confidence = mean_confidence(lines)
        shown = f", confiance moyenne {confidence:.2f}" if confidence is not None else ""
        print(f"  transcription : {time.perf_counter() - start:.0f} s{shown}")

    segmentation = dataclasses.replace(segmentation, imagename=path.name)
    page_xml = serialize(segmentation, image_size=image.size, template="pagexml")
    (out_dir / f"{stem}.page.xml").write_text(page_xml, encoding="utf-8")

    if rec_model is not None:
        text = "\n".join(line.prediction for line in lines)
        (out_dir / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")

    draw_lines(image, lines).save(out_dir / f"{stem}.lignes.jpg", quality=85)


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="*", type=Path, help="images (défaut : les n premières)")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("-n", "--count", type=int, default=3)
    parser.add_argument("--seg-model", type=Path, default=None, help="défaut : modèle kraken")
    parser.add_argument("--rec-model", type=Path, default=None, help="modèle de transcription")
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
    if args.rec_model is None:
        print("Pas de --rec-model : segmentation seule, sans transcription.", file=sys.stderr)

    threads = os.cpu_count() or 1
    common = {"accelerator": args.accelerator, "num_threads": threads}
    seg_config = SegmentationInferenceConfig(**common)
    rec_config = RecognitionInferenceConfig(**common)

    seg_model = SegmentationTaskModel.load_model(args.seg_model)
    rec_model = RecognitionTaskModel.load_model(args.rec_model) if args.rec_model else None

    args.output.mkdir(parents=True, exist_ok=True)
    for path in images:
        print(path)
        process_page(path, seg_model, rec_model, seg_config, rec_config, args.output)
    print(f"Sorties écrites dans {args.output}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
