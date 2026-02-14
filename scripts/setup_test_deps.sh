#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/darrieythorsson/compHydro/code"

CFUSE_REPO="${CFUSE_REPO:-$ROOT/dFUSE}"
DROUTE_REPO="${DROUTE_REPO:-$ROOT/dRoute}"
DGW_REPO="${DGW_REPO:-$ROOT/dgw}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3 not found. Set PYTHON_BIN to your Python executable." >&2
  exit 1
fi

if [[ "${FORCE_PIP:-0}" == "1" ]]; then
  $PYTHON_BIN -m pip install -e "$CFUSE_REPO"
  $PYTHON_BIN -m pip install -e "$DROUTE_REPO"
else
  echo "Skipping pip installs (FORCE_PIP=1 to enable)."
  echo "Using local PYTHONPATH entries for cfuse/droute."
fi

echo "\nChecking dGW Python bindings..."
BUILD_DIR="$DGW_REPO/build"
if ls "$BUILD_DIR"/dgw_py*.so >/dev/null 2>&1; then
  echo "Found existing dgw_py module in $BUILD_DIR"
  echo "Add to PYTHONPATH if needed: $BUILD_DIR"
  exit 0
fi

mkdir -p "$BUILD_DIR"

$PYTHON_BIN - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY

SITE_PKGS="$($PYTHON_BIN - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

cmake -S "$DGW_REPO" -B "$BUILD_DIR" \
  -DDGW_ENABLE_PYTHON=ON \
  -DPython3_EXECUTABLE="$($PYTHON_BIN -c 'import sys; print(sys.executable)')" \
  -DPython3_SITELIB="$SITE_PKGS"

cmake --build "$BUILD_DIR" -j 8
cmake --install "$BUILD_DIR"

cat <<EOF
\nDone. If import still fails, ensure:
  export PYTHONPATH="$DGW_REPO/python:\$PYTHONPATH"
EOF
