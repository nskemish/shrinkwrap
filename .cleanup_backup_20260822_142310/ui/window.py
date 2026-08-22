from pathlib import Path

from PySide6.QtCore import Qt

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
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.loader import (
    MeshLoadError,
    get_mesh_info,
)

from core.heightmap import (
    generate_heightmap,
)

from core.envelope import (
    build_envelope,
    envelope_to_mesh,
)

from core.drill import (
    drill_center,
    filter_drills,
    load_excellon,
    transform_drills,
)

from core.export import (
    export_mesh,
)

from ui.viewport import Viewport


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "ShrinkWrap"
        )

        self.resize(
            1450,
            900,
        )

        self.viewport = Viewport()

        self.generated_mesh = None

        self.npth_path = None

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
            0, 0, 0, 0
        )

        layout.setSpacing(
            0
        )

        sidebar = self._create_sidebar()

        scroll = QScrollArea()
        scroll.setWidget(sidebar)
        scroll.setWidgetResizable(True)

        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )

        scroll.setFixedWidth(350)
        scroll.setFrameShape(QFrame.NoFrame)

        scroll.setStyleSheet(
            """
            QScrollArea {
                background: #14171c;
                border: none;
            }

            QScrollBar:vertical {
                background: #14171c;
                width: 10px;
            }

            QScrollBar::handle:vertical {
                background: #39414c;
                min-height: 30px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical:hover {
                background: #4a5563;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )

        layout.addWidget(scroll)

        layout.addWidget(
            self.viewport,
            1,
        )

    def _create_sidebar(self):

        sidebar = QFrame()

        sidebar.setMinimumWidth(
            330
        )

        sidebar.setObjectName(
            "sidebar"
        )

        layout = QVBoxLayout(
            sidebar
        )

        layout.setContentsMargins(
            20, 20, 20, 20
        )

        layout.setSpacing(
            12
        )

        # =====================================================
        # HEADER
        # =====================================================

        title = QLabel(
            "SHRINKWRAP"
        )

        title.setObjectName(
            "title"
        )

        subtitle = QLabel(
            "PCB envelope + Excellon drilling"
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

        # =====================================================
        # MODEL
        # =====================================================

        model_group = QGroupBox(
            "3D Model"
        )

        model_layout = QVBoxLayout(
            model_group
        )

        self.model_label = QLabel(
            "No model loaded"
        )

        self.model_label.setWordWrap(
            True
        )

        open_model_button = QPushButton(
            "Open model..."
        )

        open_model_button.clicked.connect(
            self.open_model
        )

        focus_button = QPushButton(
            "Focus model"
        )

        focus_button.clicked.connect(
            self.viewport.focus_mesh
        )

        model_layout.addWidget(
            self.model_label
        )

        model_layout.addWidget(
            open_model_button
        )

        model_layout.addWidget(
            focus_button
        )

        layout.addWidget(
            model_group
        )

        # =====================================================
        # DRILL FILES
        # =====================================================

        drill_group = QGroupBox(
            "Drill Data"
        )

        drill_layout = QVBoxLayout(
            drill_group
        )

        self.npth_label = QLabel(
            "NPTH: not loaded"
        )

        self.npth_label.setWordWrap(
            True
        )

        npth_button = QPushButton(
            "Load NPTH.drl..."
        )

        npth_button.clicked.connect(
            self.open_npth
        )

        drill_layout.addWidget(
            self.npth_label
        )

        drill_layout.addWidget(
            npth_button
        )

        layout.addWidget(
            drill_group
        )

        # =====================================================
        # DRILL SETTINGS
        # =====================================================

        drill_settings = QGroupBox(
            "Drill Settings"
        )

        drill_form = QFormLayout(
            drill_settings
        )

        self.min_drill_spin = (
            QDoubleSpinBox()
        )

        self.min_drill_spin.setRange(
            0.0,
            20.0,
        )

        self.min_drill_spin.setDecimals(
            3
        )

        self.min_drill_spin.setValue(
            2.50
        )

        self.min_drill_spin.setSingleStep(
            0.10
        )

        self.min_drill_spin.setSuffix(
            " mm"
        )

        self.drill_offset_x = (
            QDoubleSpinBox()
        )

        self.drill_offset_x.setRange(
            -1000.0,
            1000.0,
        )

        self.drill_offset_x.setDecimals(
            3
        )

        self.drill_offset_x.setSuffix(
            " mm"
        )

        self.drill_offset_y = (
            QDoubleSpinBox()
        )

        self.drill_offset_y.setRange(
            -1000.0,
            1000.0,
        )

        self.drill_offset_y.setDecimals(
            3
        )

        self.drill_offset_y.setSuffix(
            " mm"
        )

        self.flip_x_checkbox = (
            QCheckBox(
                "Flip X"
            )
        )

        self.flip_y_checkbox = (
            QCheckBox(
                "Flip Y"
            )
        )

        auto_center_button = QPushButton(
            "Auto-center drill coordinates"
        )

        auto_center_button.clicked.connect(
            self.auto_center_drills
        )

        drill_form.addRow(
            "Minimum Ø",
            self.min_drill_spin,
        )

        self.hole_keepout_spin = (
            QDoubleSpinBox()
        )

        self.hole_keepout_spin.setRange(
            0.0,
            20.0,
        )

        self.hole_keepout_spin.setDecimals(
            2
        )

        self.hole_keepout_spin.setValue(
            2.0
        )

        self.hole_keepout_spin.setSingleStep(
            0.25
        )

        self.hole_keepout_spin.setSuffix(
            " mm"
        )

        drill_form.addRow(
            "Wrap keepout",
            self.hole_keepout_spin,
        )


        drill_form.addRow(
            "X offset",
            self.drill_offset_x,
        )

        drill_form.addRow(
            "Y offset",
            self.drill_offset_y,
        )

        drill_form.addRow(
            self.flip_x_checkbox
        )

        drill_form.addRow(
            self.flip_y_checkbox
        )

        drill_form.addRow(
            auto_center_button
        )

        layout.addWidget(
            drill_settings
        )

        # =====================================================
        # ENVELOPE
        # =====================================================

        envelope_group = QGroupBox(
            "Envelope"
        )

        form = QFormLayout(
            envelope_group
        )

        self.resolution_spin = (
            QDoubleSpinBox()
        )

        self.resolution_spin.setRange(
            0.05,
            5.0,
        )

        self.resolution_spin.setValue(
            0.20
        )

        self.resolution_spin.setSingleStep(
            0.05
        )

        self.resolution_spin.setSuffix(
            " mm"
        )

        self.bridge_spin = (
            QDoubleSpinBox()
        )

        self.bridge_spin.setRange(
            0.0,
            50.0,
        )

        self.bridge_spin.setValue(
            4.0
        )

        self.bridge_spin.setSingleStep(
            0.5
        )

        self.bridge_spin.setSuffix(
            " mm"
        )

        self.slope_spin = (
            QDoubleSpinBox()
        )

        self.slope_spin.setRange(
            5.0,
            88.0,
        )

        self.slope_spin.setValue(
            55.0
        )

        self.slope_spin.setSingleStep(
            5.0
        )

        self.slope_spin.setSuffix(
            "°"
        )

        self.smoothing_spin = (
            QDoubleSpinBox()
        )

        self.smoothing_spin.setRange(
            0.0,
            20.0,
        )

        self.smoothing_spin.setValue(
            0.40
        )

        self.smoothing_spin.setSingleStep(
            0.10
        )

        self.smoothing_spin.setSuffix(
            " mm"
        )

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
            0.10
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

        # =====================================================
        # DISPLAY
        # =====================================================

        display_group = QGroupBox(
            "Display"
        )

        display_layout = QVBoxLayout(
            display_group
        )

        self.wireframe_checkbox = (
            QCheckBox(
                "Wireframe"
            )
        )

        self.wireframe_checkbox.toggled.connect(
            self.viewport.set_wireframe
        )

        show_original = QPushButton(
            "Show original"
        )

        show_original.clicked.connect(
            self.show_original
        )

        self.show_wrap_button = QPushButton(
            "Show wrap"
        )

        self.show_wrap_button.setEnabled(
            False
        )

        self.show_wrap_button.clicked.connect(
            self.show_wrap
        )

        display_layout.addWidget(
            self.wireframe_checkbox
        )

        display_layout.addWidget(
            show_original
        )

        display_layout.addWidget(
            self.show_wrap_button
        )

        layout.addWidget(
            display_group
        )

        # =====================================================
        # GENERATE / EXPORT
        # =====================================================

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
            self.generate_button
        )

        layout.addWidget(
            self.export_button
        )

        layout.addStretch()

        self._apply_style(
            sidebar
        )

        return sidebar

    # =========================================================
    # FILES
    # =========================================================

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

            mesh = self.viewport.open_mesh(
                path
            )

        except MeshLoadError as exc:

            self.statusBar().showMessage(
                str(exc)
            )

            return

        self.generated_mesh = None

        self.export_button.setEnabled(
            False
        )

        self.show_wrap_button.setEnabled(
            False
        )

        info = get_mesh_info(
            mesh
        )

        self.model_label.setText(
            Path(path).name
        )

        self.statusBar().showMessage(
            (
                f"{Path(path).name} | "
                f"{info['width']:.2f} × "
                f"{info['depth']:.2f} × "
                f"{info['height']:.2f} mm | "
                f"{info['triangles']:,} triangles"
            )
        )

    def open_npth(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open NPTH drill file",
            "",
            "Excellon Drill (*.drl *.xln *.txt);;All files (*)",
        )

        if not path:
            return

        try:

            holes = load_excellon(
                path,
                plated=False,
            )

        except Exception as exc:

            self.statusBar().showMessage(
                f"NPTH error: {exc}"
            )

            raise

        self.npth_path = path

        self.npth_label.setText(
            (
                f"NPTH: {Path(path).name}\n"
                f"{len(holes)} drill positions"
            )
        )

    # =========================================================
    # DRILLS
    # =========================================================

    def _raw_selected_drills(self):
        """
        ShrinkWrap koristi SAMO NPTH.

        PTH/vias/THT drill rupe namerno ignorišemo:
        rigid PCB reconstruction ih automatski popunjava.
        """

        if not self.npth_path:
            return []

        holes = load_excellon(
            self.npth_path,
            plated=False,
        )

        #
        # Minimum diameter i dalje može biti koristan
        # ako NPTH fajl sadrži neke sitne mehaničke otvore.
        #
        holes = filter_drills(
            holes,

            min_diameter_mm=(
                self.min_drill_spin.value()
            ),
        )

        return holes

    def _transformed_drills(self):

        holes = (
            self._raw_selected_drills()
        )

        if not holes:
            return []

        center = drill_center(
            holes
        )

        if center is None:
            center_x = 0.0
            center_y = 0.0
        else:
            center_x, center_y = (
                center
            )

        return transform_drills(
            holes,

            offset_x=(
                self.drill_offset_x.value()
            ),

            offset_y=(
                self.drill_offset_y.value()
            ),

            flip_x=(
                self.flip_x_checkbox.isChecked()
            ),

            flip_y=(
                self.flip_y_checkbox.isChecked()
            ),

            center_x=center_x,
            center_y=center_y,
        )

    def auto_center_drills(self):

        if self.viewport.mesh is None:

            self.statusBar().showMessage(
                "Load the 3D model first."
            )

            return

        holes = (
            self._raw_selected_drills()
        )

        if not holes:

            self.statusBar().showMessage(
                "No enabled drill data."
            )

            return

        drill_mid = drill_center(
            holes
        )

        if drill_mid is None:
            return

        mesh_bounds = (
            self.viewport.mesh.bounds
        )

        model_center_x = (
            mesh_bounds[0][0]
            +
            mesh_bounds[1][0]
        ) * 0.5

        model_center_y = (
            mesh_bounds[0][1]
            +
            mesh_bounds[1][1]
        ) * 0.5

        drill_center_x, drill_center_y = (
            drill_mid
        )

        self.drill_offset_x.setValue(
            model_center_x
            -
            drill_center_x
        )

        self.drill_offset_y.setValue(
            model_center_y
            -
            drill_center_y
        )

        self.statusBar().showMessage(
            (
                "Drill coordinates centered. "
                "Fine-tune X/Y offsets if necessary."
            )
        )

    # =========================================================
    # GENERATE
    # =========================================================

    def generate_wrap(self):

        if self.viewport.mesh is None:

            self.statusBar().showMessage(
                "Load a model first."
            )

            return

        try:

            #
            # ------------------------------------------------
            # NPTH
            # ------------------------------------------------
            #

            holes = (
                self._transformed_drills()
            )

            #
            # ------------------------------------------------
            # TOP / BOTTOM PROJECTION + PCB DETECTION
            # ------------------------------------------------
            #

            self.statusBar().showMessage(
                "Projecting model and detecting rigid PCB..."
            )

            raw = generate_heightmap(
                mesh=self.viewport.mesh,

                resolution=(
                    self.resolution_spin.value()
                ),
            )

            #
            # ------------------------------------------------
            # COMPONENT CLOTH
            # ------------------------------------------------
            #

            self.statusBar().showMessage(
                (
                    f"PCB detected: "
                    f"{raw.pcb_thickness:.3f} mm — "
                    f"building component envelope..."
                )
            )

            envelope = build_envelope(
                heightmap=raw,

                smoothing_mm=(
                    self.smoothing_spin.value()
                ),

                clearance_mm=(
                    self.clearance_spin.value()
                ),

                bridge_mm=(
                    self.bridge_spin.value()
                ),

                slope_degrees=(
                    self.slope_spin.value()
                ),

                holes=holes,

                hole_keepout_mm=(
                    self.hole_keepout_spin.value()
                ),
            )

            #
            # ------------------------------------------------
            # FINAL SOLID
            # ------------------------------------------------
            #

            self.statusBar().showMessage(
                (
                    f"Building rigid PCB + cloth "
                    f"with {len(holes)} NPTH holes..."
                )
            )

            mesh = envelope_to_mesh(
                envelope
            )

            self.generated_mesh = mesh

            self.viewport.show_generated_mesh(
                mesh
            )

            self.export_button.setEnabled(
                True
            )

            self.show_wrap_button.setEnabled(
                True
            )

            self.statusBar().showMessage(
                (
                    f"Wrap generated | "
                    f"PCB {raw.pcb_thickness:.3f} mm | "
                    f"{len(holes)} NPTH | "
                    f"{len(mesh.vertices):,} vertices | "
                    f"{len(mesh.faces):,} triangles"
                )
            )

        except Exception as exc:

            self.statusBar().showMessage(
                f"Generation error: {exc}"
            )

            raise

    # =========================================================
    # DISPLAY / EXPORT
    # =========================================================

    def show_original(self):

        self.viewport.show_original_mesh()

    def show_wrap(self):

        if self.generated_mesh is not None:

            self.viewport.show_generated_mesh(
                self.generated_mesh
            )

    def export_generated(self):

        if self.generated_mesh is None:
            return

        path, selected = (
            QFileDialog.getSaveFileName(
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
        )

        if not path:
            return

        if not Path(
            path
        ).suffix:

            if "OBJ" in selected:
                path += ".obj"

            elif "PLY" in selected:
                path += ".ply"

            elif "GLB" in selected:
                path += ".glb"

            else:
                path += ".stl"

        export_mesh(
            self.generated_mesh,
            path,
        )

        self.statusBar().showMessage(
            f"Exported: {path}"
        )

    # =========================================================
    # STYLE
    # =========================================================

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

            QPushButton#primaryButton {
                background-color: #356f54;
                border-color: #478c6b;
                font-weight: 600;
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