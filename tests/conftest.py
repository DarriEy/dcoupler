from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest


def _try_add_repo_path(env_var: str, fallback: str, package_subdir: str = "python") -> None:
    repo = os.environ.get(env_var, fallback)
    repo_path = Path(repo)
    if not repo_path.exists():
        return
    candidate = repo_path / package_subdir
    if candidate.exists():
        sys.path.insert(0, str(candidate))


def _try_add_path(path: str) -> None:
    candidate = Path(path)
    if candidate.exists():
        sys.path.insert(0, str(candidate))


def module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def require_modules(*names: str) -> None:
    missing = [name for name in names if not module_available(name)]
    if missing:
        pytest.skip(f"Missing modules: {', '.join(missing)}")


def pytest_configure(config):
    root = "/Users/darrieythorsson/compHydro/code"
    extra = os.environ.get("EXTRA_SITE_PACKAGES")
    if extra:
        _try_add_path(extra)
    else:
        _try_add_path(f"{root}/SYMFLUENCE/.venv/lib/python3.11/site-packages")
    _try_add_repo_path("CFUSE_REPO", f"{root}/dFUSE")
    _try_add_repo_path("DROUTE_REPO", f"{root}/dRoute")
    _try_add_repo_path("DGW_REPO", f"{root}/dgw")
    _try_add_path(os.environ.get("DGW_BUILD", f"{root}/dgw/build"))
