from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.loader import (
    MeshLoadError,
    get_mesh_info,
)

from core.heightmap import generate_heightmap

from core.envelope import (
    build_envelope,
    envelope_to_mesh,
)

from core.export import export_mesh

from ui.viewport import Viewport


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "ShrinkWrap"
        )

        self.resize(
            1400,
            850,
        )

        self.viewport = Viewport()

        self.generated_mesh = None

        self._build_ui()

        self.statusBar().showMessage(
            "Ready"
        )

    def _build_ui(self):

        root = QWidget()

        self.setCentralWidget(
            root
        )

        layout = QHBoxLayout(
            root
        )

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        layout.setSpacing(
            0
        )

        sidebar = (
            self._create_sidebar()
        )

        layout.addWidget(
            sidebar
        )

        layout.addWidget(
            self.viewport,
            1,
        )

    def _create_sidebar(self) -> QWidget:

        sidebar = QFrame()

        sidebar.setFixedWidth(
            300
        )

        sidebar.setObjectName(
            "sidebar"
        )

        layout = QVBoxLayout(
            sidebar
        )

        layout.setContentsMargins(
            20,
            20,
            20,
            20,
        )

        layout.setSpacing(
            14
        )

        #
        # Header
        #

        title = QLabel(
            "SHRINKWRAP"
        )

        title.setObjectName(
            "title"
        )

        subtitle = QLabel(
            "PCB envelope generator"
        )

        subtitle.setObjectName(
            "subtitle"
        )

        layout.addWidget(
            title
        )

        layout.addWidget(
            subtitle
        )

        #
        # Input
        #

        input_group = QGroupBox(
            "Input"
        )

        input_layout = QVBoxLayout(
            input_group
        )

        self.file_label = QLabel(
            "No model loaded"
        )

        self.file_label.setWordWrap(
            True
        )

        open_button = QPushButton(
            "Open model..."
        )

        open_button.clicked.connect(
            self.open_model
        )

        focus_button = QPushButton(
            "Focus model"
        )

        focus_button.clicked.connect(
            self.viewport.focus_mesh
        )

        input_layout.addWidget(
            self.file_label
        )

        input_layout.addWidget(
            open_button
        )

        input_layout.addWidget(
            focus_button
        )

        layout.addWidget(
            input_group
        )

        #
        # Envelope settings
        #

        envelope_group = QGroupBox(
            "Envelope"
        )

        form = QFormLayout(
            envelope_group
        )

        #
        # Resolution
        #

        self.resolution_spin = (
            QDoubleSpinBox()
        )

        self.resolution_spin.setRange(
            0.05,
            5.0,
        )

        self.resolution_spin.setValue(
            0.30
        )

        self.resolution_spin.setSingleStep(
            0.05
        )

        self.resolution_spin.setSuffix(
            " mm"
        )

        #
        # Bridge
        #

        self.bridge_spin = (
            QDoubleSpinBox()
        )

        self.bridge_spin.setRange(
            0.0,
            50.0,
        )

        self.bridge_spin.setValue(
            6.0
        )

        self.bridge_spin.setSingleStep(
            0.5
        )

        self.bridge_spin.setSuffix(
            " mm"
        )

        #
        # Cloth slope
        #

        self.slope_spin = (
            QDoubleSpinBox()
        )

        self.slope_spin.setRange(
            5.0,
            85.0,
        )

        self.slope_spin.setValue(
            45.0
        )

        self.slope_spin.setSingleStep(
            5.0
        )

        self.slope_spin.setSuffix(
            "°"
        )

        #
        # Smoothing
        #

        self.smoothing_spin = (
            QDoubleSpinBox()
        )

        self.smoothing_spin.setRange(
            0.0,
            20.0,
        )

        self.smoothing_spin.setValue(
            0.4
        )

        self.smoothing_spin.setSingleStep(
            0.1
        )

        self.smoothing_spin.setSuffix(
            " mm"
        )

        #
        # Clearance
        #

        self.clearance_spin = (
            QDoubleSpinBox()
        )

        self.clearance_spin.setRange(
            0.0,
            20.0,
        )

        self.clearance_spin.setValue(
            0.0
        )

        self.clearance_spin.setSingleStep(
            0.1
        )

        self.clearance_spin.setSuffix(
            " mm"
        )

        form.addRow(
            "Resolution",
            self.resolution_spin,
        )

        form.addRow(
            "Bridge gaps",
            self.bridge_spin,
        )

        form.addRow(
            "Cloth slope",
            self.slope_spin,
        )

        form.addRow(
            "Smoothing",
            self.smoothing_spin,
        )

        form.addRow(
            "Clearance",
            self.clearance_spin,
        )

        layout.addWidget(
            envelope_group
        )

        #
        # Display
        #

        display_group = QGroupBox(
            "Display"
        )

        display_layout = QVBoxLayout(
            display_group
        )

        self.wireframe_checkbox = QCheckBox(
            "Wireframe"
        )

        self.wireframe_checkbox.toggled.connect(
            self.viewport.set_wireframe
        )

        self.original_button = QPushButton(
            "Show original"
        )

        self.original_button.clicked.connect(
            self.show_original
        )

        self.wrap_button = QPushButton(
            "Show wrap"
        )

        self.wrap_button.clicked.connect(
            self.show_wrap
        )

        self.wrap_button.setEnabled(
            False
        )

        display_layout.addWidget(
            self.wireframe_checkbox
        )

        display_layout.addWidget(
            self.original_button
        )

        display_layout.addWidget(
            self.wrap_button
        )

        layout.addWidget(
            display_group
        )

        #
        # Generate
        #

        self.generate_button = QPushButton(
            "Generate Wrap"
        )

        self.generate_button.setObjectName(
            "primaryButton"
        )

        self.generate_button.setMinimumHeight(
            44
        )

        self.generate_button.clicked.connect(
            self.generate_wrap
        )

        layout.addWidget(
            self.generate_button
        )

        #
        # Export
        #

        self.export_button = QPushButton(
            "Export STL"
        )

        self.export_button.setEnabled(
            False
        )

        self.export_button.clicked.connect(
            self.export_generated
        )

        layout.addWidget(
            self.export_button
        )

        layout.addStretch()

        controls = QLabel(
            "LEFT DRAG   Orbit\n"
            "RIGHT DRAG  Pan\n"
            "MIDDLE      Pan\n"
            "SCROLL      Zoom"
        )

        controls.setObjectName(
            "controls"
        )

        layout.addWidget(
            controls
        )

        self._apply_style(
            sidebar
        )

        return sidebar

    def _apply_style(
        self,
        widget,
    ):

        widget.setStyleSheet(
            """
            #sidebar {
                background-color: #14171c;
                color: #e8ebef;
            }

            QLabel {
                color: #e8ebef;
            }

            QLabel#title {
                font-size: 23px;
                font-weight: 700;
            }

            QLabel#subtitle {
                color: #89919d;
                margin-bottom: 8px;
            }

            QLabel#controls {
                color: #747d89;
                font-family: monospace;
                font-size: 11px;
            }

            QGroupBox {
                color: #dce1e8;
                border: 1px solid #303640;
                border-radius: 7px;
                margin-top: 10px;
                padding-top: 12px;
                font-weight: 600;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }

            QPushButton {
                background-color: #252b34;
                color: #edf0f4;
                border: 1px solid #39414c;
                border-radius: 6px;
                padding: 8px;
            }

            QPushButton:hover {
                background-color: #303844;
            }

            QPushButton:pressed {
                background-color: #1e232a;
            }

            QPushButton#primaryButton {
                background-color: #356f54;
                border-color: #478c6b;
                font-weight: 600;
            }

            QPushButton#primaryButton:hover {
                background-color: #3d7d60;
            }

            QPushButton:disabled {
                color: #626a75;
                background-color: #1b1e23;
                border-color: #292e35;
            }

            QDoubleSpinBox {
                background-color: #1d2128;
                color: #e8ebef;
                border: 1px solid #353c46;
                border-radius: 5px;
                padding: 4px;
            }

            QCheckBox {
                color: #dfe3e8;
            }
            """
        )

    def open_model(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open PCB model",
            "",
            (
                "3D Mesh "
                "(*.stl *.obj *.ply *.glb *.gltf);;"
                "All files (*)"
            ),
        )

        if not path:
            return

        try:

            mesh = (
                self.viewport.open_mesh(
                    path
                )
            )

        except MeshLoadError as exc:

            self.statusBar().showMessage(
                str(exc)
            )

            return

        except Exception as exc:

            self.statusBar().showMessage(
                f"OpenGL error: {exc}"
            )

            raise

        self.generated_mesh = None

        self.export_button.setEnabled(
            False
        )

        self.wrap_button.setEnabled(
            False
        )

        info = get_mesh_info(
            mesh
        )

        self.file_label.setText(
            Path(path).name
        )

        self.statusBar().showMessage(
            (
                f"{Path(path).name}    |    "
                f"{info['width']:.2f} × "
                f"{info['depth']:.2f} × "
                f"{info['height']:.2f} mm    |    "
                f"{info['triangles']:,} triangles"
            )
        )

    def generate_wrap(self):

        if self.viewport.mesh is None:

            self.statusBar().showMessage(
                "Load a model first."
            )

            return

        resolution = (
            self.resolution_spin.value()
        )

        bridge = (
            self.bridge_spin.value()
        )

        slope = (
            self.slope_spin.value()
        )

        smoothing = (
            self.smoothing_spin.value()
        )

        clearance = (
            self.clearance_spin.value()
        )

        try:

            self.statusBar().showMessage(
                "Generating height map..."
            )

            #
            # Mesh -> raw height map
            #

            raw_heightmap = (
                generate_heightmap(
                    mesh=self.viewport.mesh,
                    resolution=resolution,
                )
            )

            self.statusBar().showMessage(
                "Generating cloth envelope..."
            )

            #
            # Raw height map -> cloth envelope
            #

            envelope = (
                build_envelope(
                    heightmap=raw_heightmap,
                    smoothing_mm=smoothing,
                    clearance_mm=clearance,
                    bridge_mm=bridge,
                    slope_degrees=slope,
                )
            )

            self.statusBar().showMessage(
                "Generating triangle mesh..."
            )

            #
            # Height map -> triangle mesh
            #

            mesh = envelope_to_mesh(
                envelope
            )

            self.generated_mesh = mesh

            #
            # Show generated mesh
            #

            self.viewport.show_generated_mesh(
                mesh
            )

            self.export_button.setEnabled(
                True
            )

            self.wrap_button.setEnabled(
                True
            )

            self.statusBar().showMessage(
                (
                    f"Wrap generated    |    "
                    f"{len(mesh.vertices):,} vertices    |    "
                    f"{len(mesh.faces):,} triangles"
                )
            )

        except Exception as exc:

            self.statusBar().showMessage(
                f"Generation error: {exc}"
            )

            raise

    def show_original(self):

        if self.viewport.mesh is None:
            return

        self.viewport.show_original_mesh()

        self.statusBar().showMessage(
            "Showing original model"
        )

    def show_wrap(self):

        if self.generated_mesh is None:
            return

        self.viewport.show_generated_mesh(
            self.generated_mesh
        )

        self.statusBar().showMessage(
            "Showing generated wrap"
        )

    def export_generated(self):

        if self.generated_mesh is None:
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export ShrinkWrap",
            "shrinkwrap.stl",
            (
                "STL (*.stl);;"
                "OBJ (*.obj);;"
                "PLY (*.ply);;"
                "GLB (*.glb)"
            ),
        )

        if not path:
            return

        path_obj = Path(path)

        #
        # Ako korisnik nije uneo ekstenziju,
        # dodaj je prema izabranom filteru.
        #

        if not path_obj.suffix:

            if "OBJ" in selected_filter:
                path += ".obj"

            elif "PLY" in selected_filter:
                path += ".ply"

            elif "GLB" in selected_filter:
                path += ".glb"

            else:
                path += ".stl"

        try:

            export_mesh(
                self.generated_mesh,
                path,
            )

            self.statusBar().showMessage(
                f"Exported: {path}"
            )

        except Exception as exc:

            self.statusBar().showMessage(
                f"Export error: {exc}"
            )

            raise