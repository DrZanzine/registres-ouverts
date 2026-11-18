# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ligne de commande du module ``segment``.

Usage :

    python -m registres_ouverts.segment postprocess ENTREE.json [-o SORTIE.json]

``postprocess`` lit une segmentation brute (format ``segmentation_brute.json`` : ``image``,
``taille``, ``lignes`` avec ``id``, ``baseline``, ``boundary``) et écrit la segmentation
post-traitée (``segmentation.json`` : lignes triées par ``ordre``, avec ``zone``, ``acte``,
``bbox``, et ``rejetees``). Sans ``-o``, le résultat est écrit sur la sortie standard.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from registres_ouverts.segment.postprocess import postprocess


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(
        prog="python -m registres_ouverts.segment", description="Module de segmentation."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    pp = sub.add_parser("postprocess", help="post-traiter une segmentation brute (JSON)")
    pp.add_argument("input", type=Path, help="segmentation brute (JSON)")
    pp.add_argument("-o", "--output", type=Path, default=None, help="défaut : sortie standard")
    args = parser.parse_args(argv)

    data = json.loads(args.input.read_text("utf-8"))
    result = postprocess(data)
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.output is None:
        sys.stdout.write(text + "\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", "utf-8")
        zones = Counter(ln["zone"] for ln in result["lignes"])
        actes = len({ln["acte"] for ln in result["lignes"] if ln["acte"] is not None})
        print(
            f"{len(result['lignes'])} ligne(s) conservée(s), {len(result['rejetees'])} rejetée(s), "
            f"{actes} acte(s) ; zones : {dict(zones)}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
