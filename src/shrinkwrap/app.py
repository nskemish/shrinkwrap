import sys

from PySide6.QtGui import QSurfaceFormat, QFontDatabase
from PySide6.QtWidgets import QApplication

from shrinkwrap.ui.window import MainWindow
from shrinkwrap.resources import resources_rc


def main():
    fmt = QSurfaceFormat()

    # macOS podržava maksimum OpenGL 4.1 Core.
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.CoreProfile)

    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(4)

    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)

    app.setApplicationName("ShrinkWrap")
    app.setOrganizationName("ShrinkWrap")

    font_id = QFontDatabase.addApplicationFont(
        ":/branding/font.ttf"
    )

    if font_id < 0:
        print(
            "[font] Failed to load embedded font"
        )
    else:
        families = (
            QFontDatabase.applicationFontFamilies(
                font_id
            )
        )

        if families:
            font = app.font()
            font.setFamily(
                families[0]
            )
            app.setFont(
                font
            )

            print(
                "[font] Using:",
                families[0],
            )

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())