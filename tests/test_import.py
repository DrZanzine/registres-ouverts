# Copyright (C) 2026 Ziane Menni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vérifie que le paquet et ses huit modules s'importent."""

import importlib

import pytest

MODULES = [
    "acquire",
    "segment",
    "transcribe",
    "extract",
    "check",
    "review",
    "store",
    "evaluate",
]


def test_package_imports() -> None:
    """Le paquet racine s'importe et expose sa version."""
    package = importlib.import_module("registres_ouverts")
    assert isinstance(package.__version__, str)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    """Chaque module de la chaîne s'importe."""
    importlib.import_module(f"registres_ouverts.{name}")
