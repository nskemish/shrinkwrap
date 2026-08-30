import sys

from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from shrinkwrap.ui.window import MainWindow


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

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())