# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

from PyInstaller.utils.hooks import (
    collect_dynamic_libs,
    collect_submodules,
)

PROJECT_ROOT = Path(SPECPATH).resolve()

# ============================================================
# NATIVE SHRINKWRAP MODULE
# ============================================================

native_candidates = sorted(
    (PROJECT_ROOT / "core").glob("_native*.so")
)

if not native_candidates:
    raise RuntimeError(
        "\n\n"
        "ShrinkWrap native C++ module was not found.\n"
        "Build it first with CMake before running PyInstaller.\n\n"
        "Expected something like:\n"
        "    core/_native.cpython-314-darwin.so\n"
    )

NATIVE_MODULE = native_candidates[0]

print()
print("=" * 72)
print("ShrinkWrap PyInstaller")
print("=" * 72)
print(f"Project:       {PROJECT_ROOT}")
print(f"Native module: {NATIVE_MODULE}")
print()


# ============================================================
# DATA FILES
# ============================================================

datas = [
    (
        str(PROJECT_ROOT / "shaders"),
        "shaders",
    ),
]


# ============================================================
# BINARIES
# ============================================================

#
# Explicitly bundle our pybind11 extension.
#
# PyInstaller will inspect its Mach-O dependencies and pull in
# required non-system dynamic libraries where applicable.
#
binaries = [
    (
        str(NATIVE_MODULE),
        "core",
    ),
]


# ============================================================
# HIDDEN IMPORTS
# ============================================================

hiddenimports = [
    "core._native",

    # PyOpenGL dynamically resolves quite a bit at runtime.
    "OpenGL",
    "OpenGL.GL",
    "OpenGL.arrays",
    "OpenGL.arrays.vbo",

    # Keep the important SciPy namespaces explicit.
    "scipy",
    "scipy.ndimage",

    # trimesh uses lazy/dynamic imports in several places.
    "trimesh",
]

hiddenimports += collect_submodules(
    "OpenGL"
)


# ============================================================
# ANALYSIS
# ============================================================

a = Analysis(
    [
        str(PROJECT_ROOT / "main.py"),
    ],

    pathex=[
        str(PROJECT_ROOT),
    ],

    binaries=binaries,
    datas=datas,

    hiddenimports=hiddenimports,

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    excludes=[
        "pytest",
        "IPython",
        "jupyter",
        "matplotlib",
    ],

    noarchive=False,
    optimize=1,
)


# ============================================================
# PYZ
# ============================================================

pyz = PYZ(
    a.pure
)


# ============================================================
# EXECUTABLE
# ============================================================

exe = EXE(
    pyz,
    a.scripts,
    [],

    exclude_binaries=True,

    name="ShrinkWrap",

    debug=False,
    bootloader_ignore_signals=False,

    strip=False,
    upx=False,

    console=False,

    argv_emulation=False,

    target_arch="arm64",
)


# ============================================================
# COLLECT
# ============================================================

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,

    strip=False,
    upx=False,

    name="ShrinkWrap",
)


# ============================================================
# macOS APP BUNDLE
# ============================================================

app = BUNDLE(
    coll,

    name="ShrinkWrap.app",

    icon=None,

    bundle_identifier="com.shrinkwrap.app",

    info_plist={
        "CFBundleName": "ShrinkWrap",
        "CFBundleDisplayName": "ShrinkWrap",

        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "1",

        "NSHighResolutionCapable": True,

        # Qt/OpenGL application.
        "LSMinimumSystemVersion": "13.0",

        "NSPrincipalClass": "NSApplication",
    },
)
