from __future__ import annotations

import pytest

from tests.conftest import require_modules


def test_cfuse_import():
    require_modules("cfuse")
    import cfuse  # noqa: F401


def test_droute_import():
    require_modules("droute")
    import droute  # noqa: F401


def test_dgw_import():
    require_modules("dgw")
    import dgw  # noqa: F401
