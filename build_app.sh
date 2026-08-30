#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

echo
echo "============================================================"
echo " ShrinkWrap — native build"
echo "============================================================"
echo

PYTHON_EXECUTABLE="$(
    uv run python -c 'import sys; print(sys.executable)'
)"

PYBIND11_DIR="$(
    uv run python -m pybind11 --cmakedir
)"

cmake \
    -S . \
    -B build/native \
    -G Ninja \
    -DPython_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -Dpybind11_DIR="$PYBIND11_DIR" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build \
    build/native \
    --parallel


echo
echo "============================================================"
echo " ShrinkWrap — native smoke test"
echo "============================================================"
echo

uv run python - <<'PY'
from core import _native

print("Native module:", _native)
print("project_mesh:", _native.project_mesh)
print("propagate_relief:", _native.propagate_relief)
print("regularizer:", _native.regularize_steep_surface)
print("adaptive mesher:", _native.adaptive_envelope_to_mesh)
print("QEM:", _native.simplify_qem)
PY


echo
echo "============================================================"
echo " ShrinkWrap — macOS icon"
echo "============================================================"
echo

if [[ -f "icon.png" ]]; then
    echo "Generating icon.icns from icon.png..."

    rm -rf icon.iconset
    mkdir -p icon.iconset

    sips -z 16 16 \
        icon.png \
        --out icon.iconset/icon_16x16.png \
        >/dev/null

    sips -z 32 32 \
        icon.png \
        --out icon.iconset/icon_16x16@2x.png \
        >/dev/null

    sips -z 32 32 \
        icon.png \
        --out icon.iconset/icon_32x32.png \
        >/dev/null

    sips -z 64 64 \
        icon.png \
        --out icon.iconset/icon_32x32@2x.png \
        >/dev/null

    sips -z 128 128 \
        icon.png \
        --out icon.iconset/icon_128x128.png \
        >/dev/null

    sips -z 256 256 \
        icon.png \
        --out icon.iconset/icon_128x128@2x.png \
        >/dev/null

    sips -z 256 256 \
        icon.png \
        --out icon.iconset/icon_256x256.png \
        >/dev/null

    sips -z 512 512 \
        icon.png \
        --out icon.iconset/icon_256x256@2x.png \
        >/dev/null

    sips -z 512 512 \
        icon.png \
        --out icon.iconset/icon_512x512.png \
        >/dev/null

    sips -z 1024 1024 \
        icon.png \
        --out icon.iconset/icon_512x512@2x.png \
        >/dev/null

    iconutil \
        -c icns \
        icon.iconset \
        -o icon.icns

    rm -rf icon.iconset

    echo "Created: icon.icns"

elif [[ -f "icon.icns" ]]; then
    echo "icon.png not found."
    echo "Using existing icon.icns."

else
    echo "ERROR: icon.png or icon.icns not found."
    echo
    echo "Place icon.png in the project root."
    exit 1
fi


echo
echo "============================================================"
echo " ShrinkWrap — PyInstaller"
echo "============================================================"
echo

rm -rf \
    build/ShrinkWrap \
    dist/ShrinkWrap \
    dist/ShrinkWrap.app

uv run pyinstaller \
    --clean \
    --noconfirm \
    ShrinkWrap.spec


echo
echo "============================================================"
echo " ShrinkWrap — result"
echo "============================================================"
echo

if [[ ! -d "dist/ShrinkWrap.app" ]]; then
    echo "ERROR: dist/ShrinkWrap.app was not created."
    exit 1
fi

echo "Created:"
echo
echo "    dist/ShrinkWrap.app"
echo

du -sh \
    dist/ShrinkWrap.app


echo
echo "App icon:"
find \
    dist/ShrinkWrap.app/Contents/Resources \
    -name '*.icns' \
    -print


echo
echo "Native extension:"
find \
    dist/ShrinkWrap.app \
    -name '_native*.so' \
    -print


echo
echo "Shaders:"
find \
    dist/ShrinkWrap.app \
    -path '*shaders*' \
    -type f \
    -print


echo
echo "Mach-O dependencies:"
NATIVE="$(
    find \
        dist/ShrinkWrap.app \
        -name '_native*.so' \
        -print \
        -quit
)"

if [[ -n "$NATIVE" ]]; then
    otool -L "$NATIVE"
fi


echo
echo "============================================================"
echo " ShrinkWrap.app ready"
echo "============================================================"
echo
echo "    dist/ShrinkWrap.app"
echo
echo "Run with:"
echo
echo "    open dist/ShrinkWrap.app"
echo
