# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Normalisation des textes avant comparaison.

La normalisation Unicode NFC est toujours appliquée : « é » composé (U+00E9)
et « e » + accent combinant (U+0065 U+0301) sont le même caractère pour
l'évaluation. Les autres options sont désactivées par défaut, car elles
masquent des erreurs réelles ; elles doivent être choisies explicitement et
publiées avec les résultats.

Ordre d'application : NFC → casse → diacritiques → ponctuation → espaces.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Normalization:
    """Options de normalisation (NFC toujours appliquée).

    Attributs :
        casefold: ignorer la casse (``str.casefold``, « Ä » → « ä »).
        punctuation: supprimer les signes de ponctuation (catégories Unicode
            ``P*`` : ``.,;:'-«»``…). Le trait d'union et l'apostrophe sont
            supprimés, pas remplacés par une espace : « l'an » → « lan ».
        whitespace: réduire toute suite d'espaces (espaces insécables
            comprises) à une espace et supprimer les espaces en début et fin.
        diacritics: supprimer les signes diacritiques combinants
            (« é » → « e », « ç » → « c ») ; les ligatures (« œ ») sont
            conservées.
    """

    casefold: bool = False
    punctuation: bool = False
    whitespace: bool = False
    diacritics: bool = False

    def labels(self) -> list[str]:
        """Noms des normalisations appliquées, NFC en tête."""
        names = ["nfc"]
        if self.casefold:
            names.append("casse")
        if self.diacritics:
            names.append("diacritiques")
        if self.punctuation:
            names.append("ponctuation")
        if self.whitespace:
            names.append("espaces")
        return names

    def __call__(self, text: str) -> str:
        return normalize(text, self)


def normalize(text: str, options: Normalization | None = None) -> str:
    """Applique NFC puis les options demandées à ``text``."""
    options = options or Normalization()
    text = unicodedata.normalize("NFC", text)
    if options.casefold:
        # casefold peut produire des formes décomposées (ex. « İ ») : on recompose.
        text = unicodedata.normalize("NFC", text.casefold())
    if options.diacritics:
        decomposed = unicodedata.normalize("NFD", text)
        text = unicodedata.normalize(
            "NFC", "".join(c for c in decomposed if not unicodedata.combining(c))
        )
    if options.punctuation:
        text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    if options.whitespace:
        text = " ".join(text.split())
    return text
