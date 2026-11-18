# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Interface en ligne de commande du module d'évaluation.

Usage : ``python -m registres_ouverts.evaluate {cer,champs,seuil} …``

Codes de sortie : 0 succès ; 1 critère non atteint (``cer --max``) ;
2 erreur d'usage ou fichier illisible.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import TextIO

from registres_ouverts.evaluate.fields import ActError, FieldsResult, evaluate_paths
from registres_ouverts.evaluate.metrics import Score
from registres_ouverts.evaluate.normalize import Normalization
from registres_ouverts.evaluate.threshold import (
    ThresholdError,
    ThresholdPoint,
    best_threshold,
    grid,
    load_pairs,
    threshold_curve,
)
from registres_ouverts.evaluate.transcription import (
    CorpusResult,
    LineStatus,
    TranscriptionError,
    compare_paths,
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_ERROR = 2


def _pct(value: float | None) -> str:
    return "indéfini" if value is None else f"{100 * value:.2f} %"


def _visible(text: str) -> str:
    """Rend visibles espaces et chaînes vides dans les confusions."""
    return f"«{text}»" if text else "∅"


def _add_normalization(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("normalisation (NFC toujours appliquée)")
    group.add_argument("--casse", action="store_true", help="ignorer la casse")
    group.add_argument("--ponctuation", action="store_true", help="supprimer la ponctuation")
    group.add_argument(
        "--espaces", action="store_true", help="réduire les espaces multiples, rogner les bords"
    )
    group.add_argument(
        "--diacritiques", action="store_true", help="supprimer les accents et cédilles"
    )


def _normalization(args: argparse.Namespace) -> Normalization:
    return Normalization(
        casefold=args.casse,
        punctuation=args.ponctuation,
        whitespace=args.espaces,
        diacritics=args.diacritiques,
    )


def _ratio(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError(f"valeur entre 0 et 1 attendue : {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m registres_ouverts.evaluate",
        description="Évaluation : CER/WER des transcriptions, précision et rappel par "
        "champ, courbe de seuil de confiance.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cer = sub.add_parser("cer", help="CER et WER de transcriptions (texte ou PAGE XML)")
    cer.add_argument("reference", type=Path, help="fichier ou dossier de référence")
    cer.add_argument("hypothese", type=Path, help="fichier ou dossier à évaluer")
    _add_normalization(cer)
    cer.add_argument(
        "--par-ordre", action="store_true", help="apparier les lignes par ordre, jamais par id"
    )
    cer.add_argument("--lignes", action="store_true", help="afficher le détail par ligne")
    cer.add_argument(
        "--confusions", type=int, default=10, metavar="N", help="N confusions les plus fréquentes"
    )
    cer.add_argument(
        "--max", type=_ratio, metavar="TAUX", help="code de sortie 1 si le CER global dépasse TAUX"
    )
    cer.add_argument("--json", action="store_true", help="sortie JSON")

    fields = sub.add_parser("champs", help="précision, rappel et F1 par champ des actes")
    fields.add_argument("reference", type=Path, help="acte JSON ou dossier d'actes de référence")
    fields.add_argument("hypothese", type=Path, help="acte JSON ou dossier d'actes à évaluer")
    _add_normalization(fields)
    fields.add_argument(
        "--ecarts", type=int, default=20, metavar="N", help="nombre d'écarts affichés (texte)"
    )
    fields.add_argument(
        "--export-confiances",
        type=Path,
        metavar="CSV",
        help="écrire les couples (confiance, correct) pour la sous-commande seuil",
    )
    fields.add_argument("--json", action="store_true", help="sortie JSON")

    threshold = sub.add_parser("seuil", help="courbe précision / part acceptée selon le seuil")
    source = threshold.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "couples", type=Path, nargs="?", help="CSV (confiance,correct) ou JSON de couples"
    )
    source.add_argument(
        "--actes",
        type=Path,
        nargs=2,
        metavar=("REFERENCE", "HYPOTHESE"),
        help="calculer les couples à partir d'actes (comme la sous-commande champs)",
    )
    _add_normalization(threshold)
    threshold.add_argument(
        "--pas", type=float, metavar="P", help="seuils de 0 à 1 par pas de P (défaut : observés)"
    )
    threshold.add_argument(
        "--cible", type=_ratio, default=0.99, help="précision visée (défaut : 0.99)"
    )
    threshold.add_argument("--json", action="store_true", help="sortie JSON")
    return parser


# --- cer -----------------------------------------------------------------------


def _score_line(label: str, score: Score, unit: str) -> str:
    return (
        f"{label} : {_pct(score.rate)}  ({score.errors} erreurs / {score.reference_length} "
        f"{unit} ; sub. {score.counts.substitutions}, ins. {score.counts.insertions}, "
        f"sup. {score.counts.deletions})"
    )


def _print_cer(result: CorpusResult, args: argparse.Namespace, out: TextIO) -> None:
    print(f"Normalisation : {', '.join(result.normalization.labels())}", file=out)
    print(f"Pages : {len(result.pages)}", file=out)
    for name in result.missing_pages:
        print(f"  ! page sans hypothèse : {name}", file=out)
    for name in result.extra_pages:
        print(f"  ! page sans référence (non comptée) : {name}", file=out)
    print(_score_line("CER", result.cer, "caractères"), file=out)
    print(_score_line("WER", result.wer, "mots"), file=out)
    for page in result.pages:
        if len(result.pages) > 1 or args.lignes:
            print(
                f"\n[{page.name}] CER {_pct(page.cer.rate)}, WER {_pct(page.wer.rate)}, "
                f"appariement par {page.pairing}",
                file=out,
            )
        for line in page.lines:
            index = (
                line.reference_index if line.reference_index is not None else line.hypothesis_index
            )
            ident = line.reference_id or line.hypothesis_id or f"n° {(index or 0) + 1}"
            if line.status is LineStatus.MISSING:
                print(f"  ! ligne manquante dans l'hypothèse : {ident}", file=out)
            elif line.status is LineStatus.EXTRA:
                print(f"  ! ligne en trop dans l'hypothèse : {ident}", file=out)
            if args.lignes:
                print(f"  {ident:>8}  CER {_pct(line.comparison.cer.rate)}", file=out)
                print(f"            réf. : {line.comparison.reference}", file=out)
                print(f"            hyp. : {line.comparison.hypothesis}", file=out)
    confusions = result.confusions().most_common(args.confusions)
    if confusions:
        print("\nConfusions les plus fréquentes (référence → hypothèse) :", file=out)
        for (ref, hyp), count in confusions:
            print(f"  {_visible(ref)} → {_visible(hyp)} : {count}", file=out)


def run_cer(args: argparse.Namespace, out: TextIO) -> int:
    """Sous-commande ``cer``."""
    result = compare_paths(args.reference, args.hypothese, _normalization(args), args.par_ordre)
    if args.json:
        json.dump(result.to_dict(args.confusions), out, ensure_ascii=False, indent=2)
        print(file=out)
    else:
        _print_cer(result, args, out)
    rate = result.cer.rate
    if args.max is not None and (rate is None or rate > args.max):
        if not args.json:
            print(f"\nCER global supérieur au maximum autorisé ({_pct(args.max)}).", file=out)
        return EXIT_FAILED
    return EXIT_OK


# --- champs ----------------------------------------------------------------------


def _print_fields(result: FieldsResult, max_discrepancies: int, out: TextIO) -> None:
    print(f"Normalisation : {', '.join(result.normalization.labels())}", file=out)
    print(f"Actes : {len({c.act for c in result.comparisons})}", file=out)
    for name in result.missing_acts:
        print(f"  ! acte sans hypothèse : {name}", file=out)
    for name in result.extra_acts:
        print(f"  ! acte sans référence (non compté) : {name}", file=out)
    stats = result.by_field()
    width = max([len(name) for name in stats] + [len("global")])
    print(
        f"\n{'champ':<{width}}  {'précision':>10}  {'rappel':>10}  {'F1':>10}  "
        f"{'VP':>4} {'FP':>4} {'FN':>4}",
        file=out,
    )
    for name, field_stats in [*stats.items(), ("global", result.overall())]:
        print(
            f"{name:<{width}}  {_pct(field_stats.precision):>10}  {_pct(field_stats.recall):>10}"
            f"  {_pct(field_stats.f1):>10}  {field_stats.true_positives:>4} "
            f"{field_stats.false_positives:>4} {field_stats.false_negatives:>4}",
            file=out,
        )
    discrepancies = result.discrepancies()
    if discrepancies:
        print(f"\nÉcarts ({len(discrepancies)}) :", file=out)
        for c in discrepancies[:max_discrepancies]:
            print(
                f"  [{c.act}] {c.path} ({c.outcome}) : réf. {c.reference!r} / hyp. "
                f"{c.hypothesis!r}",
                file=out,
            )
        if len(discrepancies) > max_discrepancies:
            print(f"  … {len(discrepancies) - max_discrepancies} autres (voir --json)", file=out)


def _write_pairs(path: Path, pairs: list[tuple[float, bool]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["confiance", "correct"])
        writer.writerows((confidence, int(correct)) for confidence, correct in pairs)


def run_fields(args: argparse.Namespace, out: TextIO) -> int:
    """Sous-commande ``champs``."""
    result = evaluate_paths(args.reference, args.hypothese, _normalization(args))
    if args.export_confiances:
        try:
            _write_pairs(args.export_confiances, result.confidence_pairs())
        except OSError as error:
            raise ActError(f"{args.export_confiances} : {error}") from error
    if args.json:
        json.dump(result.to_dict(), out, ensure_ascii=False, indent=2)
        print(file=out)
    else:
        _print_fields(result, args.ecarts, out)
    return EXIT_OK


# --- seuil -----------------------------------------------------------------------


def _print_threshold(
    points: list[ThresholdPoint], best: ThresholdPoint | None, target: float, out: TextIO
) -> None:
    total = points[0].total if points else 0
    print(f"Valeurs : {total}", file=out)
    print(f"\n{'seuil':>8}  {'acceptées':>10}  {'couverture':>10}  {'précision':>10}", file=out)
    for point in points:
        print(
            f"{point.threshold:>8.3f}  {point.accepted:>10}  {_pct(point.coverage):>10}  "
            f"{_pct(point.precision):>10}",
            file=out,
        )
    if best is None:
        print(f"\nAucun seuil n'atteint la précision visée ({_pct(target)}).", file=out)
    else:
        print(
            f"\nSeuil retenu pour {_pct(target)} de précision : {best.threshold:.3f} "
            f"(couverture {_pct(best.coverage)}, précision {_pct(best.precision)})",
            file=out,
        )


def run_threshold(args: argparse.Namespace, out: TextIO) -> int:
    """Sous-commande ``seuil``."""
    if args.actes:
        pairs = evaluate_paths(*args.actes, _normalization(args)).confidence_pairs()
    else:
        pairs = load_pairs(args.couples)
    thresholds = grid(args.pas) if args.pas is not None else None
    points = threshold_curve(pairs, thresholds)
    best = best_threshold(points, args.cible)
    if args.json:
        data = {
            "cible": args.cible,
            "valeurs": len(pairs),
            "seuil_retenu": best.to_dict() if best else None,
            "courbe": [point.to_dict() for point in points],
        }
        json.dump(data, out, ensure_ascii=False, indent=2)
        print(file=out)
    else:
        _print_threshold(points, best, args.cible, out)
    return EXIT_OK


COMMANDS = {"cer": run_cer, "champs": run_fields, "seuil": run_threshold}


def main(argv: list[str] | None = None, out: TextIO | None = None) -> int:
    """Point d'entrée ; renvoie le code de sortie."""
    out = out or sys.stdout
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:
        return int(exit_.code or 0)
    try:
        return COMMANDS[args.command](args, out)
    except (TranscriptionError, ActError, ThresholdError) as error:
        print(f"erreur : {error}", file=sys.stderr)
        return EXIT_ERROR
