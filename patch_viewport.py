from pathlib import Path

path = Path("ui/viewport.py")

if not path.exists():
    raise SystemExit("ui/viewport.py nije pronađen.")

text = path.read_text(encoding="utf-8")

changed = False

# ------------------------------------------------------------
# 1. Dodaj self.wrap_mesh = None posle self.mesh = None
# ------------------------------------------------------------

if "self.wrap_mesh = None" not in text:
    target = "        self.mesh = None\n"

    if target not in text:
        raise SystemExit(
            "Nisam pronašao 'self.mesh = None' u Viewport.__init__."
        )

    replacement = (
        "        self.mesh = None\n"
        "        self.wrap_mesh = None\n"
    )

    text = text.replace(
        target,
        replacement,
        1,
    )

    changed = True
    print("Dodato: self.wrap_mesh = None")
else:
    print("self.wrap_mesh već postoji.")

# ------------------------------------------------------------
# 2. Dodaj show_generated_mesh i show_original_mesh
#    pre Camera sekcije ako je pronađemo.
# ------------------------------------------------------------

need_generated = "def show_generated_mesh(" not in text
need_original = "def show_original_mesh(" not in text

if need_generated or need_original:

    methods = """
    def show_generated_mesh(
        self,
        mesh,
    ):
        self.wrap_mesh = mesh

        self.makeCurrent()

        self._upload_mesh(
            mesh
        )

        self.doneCurrent()

        self.update()

    def show_original_mesh(
        self,
    ):
        if self.mesh is None:
            return

        self.makeCurrent()

        self._upload_mesh(
            self.mesh
        )

        self.doneCurrent()

        self.update()

"""

    markers = [
        "    #\n    # Camera\n    #",
        "    # Camera",
        "    def camera_position(",
    ]

    inserted = False

    for marker in markers:
        if marker in text:
            text = text.replace(
                marker,
                methods + marker,
                1,
            )
            inserted = True
            break

    if not inserted:
        raise SystemExit(
            "Nisam našao mesto za ubacivanje metoda u Viewport klasu."
        )

    changed = True
    print("Dodato: show_generated_mesh()")
    print("Dodato: show_original_mesh()")
else:
    print("Metode za original/wrap prikaz već postoje.")

# ------------------------------------------------------------
# Sačuvaj
# ------------------------------------------------------------

if changed:
    backup = path.with_suffix(".py.bak")
    backup.write_text(
        path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    path.write_text(
        text,
        encoding="utf-8",
    )

    print()
    print(f"Patchovan: {path}")
    print(f"Backup:    {backup}")
else:
    print()
    print("Nema promena — viewport.py je već patchovan.")
