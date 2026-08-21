from __future__ import annotations

import ctypes
import math
from pathlib import Path

import numpy as np

from OpenGL.GL import *

from PySide6.QtCore import QPoint, Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from core.loader import load_mesh


ROOT_DIR = Path(
    __file__
).resolve().parent.parent

SHADER_DIR = (
    ROOT_DIR / "shaders"
)


def normalize(
    vector: np.ndarray,
) -> np.ndarray:

    length = np.linalg.norm(vector)

    if length < 1e-9:
        return vector

    return vector / length


def perspective(
    fov_degrees: float,
    aspect: float,
    near: float,
    far: float,
) -> np.ndarray:

    fov = math.radians(
        fov_degrees
    )

    f = (
        1.0 /
        math.tan(
            fov / 2.0
        )
    )

    matrix = np.zeros(
        (4, 4),
        dtype=np.float32,
    )

    matrix[0, 0] = (
        f / aspect
    )

    matrix[1, 1] = f

    matrix[2, 2] = (
        (far + near) /
        (near - far)
    )

    matrix[2, 3] = (
        (2.0 * far * near) /
        (near - far)
    )

    matrix[3, 2] = -1.0

    return matrix


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:

    forward = normalize(
        target - eye
    )

    right = normalize(
        np.cross(
            forward,
            up,
        )
    )

    true_up = np.cross(
        right,
        forward,
    )

    matrix = np.eye(
        4,
        dtype=np.float32,
    )

    matrix[0, :3] = right
    matrix[1, :3] = true_up
    matrix[2, :3] = -forward

    translation = np.eye(
        4,
        dtype=np.float32,
    )

    translation[:3, 3] = -eye

    return matrix @ translation


def load_shader_source(
    name: str,
) -> str:

    path = (
        SHADER_DIR /
        name
    )

    return path.read_text(
        encoding="utf-8"
    )


def compile_shader(
    shader_type: int,
    source: str,
) -> int:

    shader = glCreateShader(
        shader_type
    )

    glShaderSource(
        shader,
        source,
    )

    glCompileShader(
        shader
    )

    success = glGetShaderiv(
        shader,
        GL_COMPILE_STATUS,
    )

    if not success:

        log = glGetShaderInfoLog(
            shader
        )

        raise RuntimeError(
            log.decode(
                "utf-8",
                errors="replace",
            )
        )

    return shader


def create_program(
    vertex_source: str,
    fragment_source: str,
) -> int:

    vertex_shader = compile_shader(
        GL_VERTEX_SHADER,
        vertex_source,
    )

    fragment_shader = compile_shader(
        GL_FRAGMENT_SHADER,
        fragment_source,
    )

    program = glCreateProgram()

    glAttachShader(
        program,
        vertex_shader,
    )

    glAttachShader(
        program,
        fragment_shader,
    )

    glLinkProgram(
        program
    )

    success = glGetProgramiv(
        program,
        GL_LINK_STATUS,
    )

    if not success:

        log = glGetProgramInfoLog(
            program
        )

        raise RuntimeError(
            log.decode(
                "utf-8",
                errors="replace",
            )
        )

    glDeleteShader(
        vertex_shader
    )

    glDeleteShader(
        fragment_shader
    )

    return program


class Viewport(
    QOpenGLWidget
):

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)

        self.setFocusPolicy(
            Qt.StrongFocus
        )

        self.setMouseTracking(
            True
        )

        self.mesh = None
        self.wrap_mesh = None

        self.shader_program = 0

        self.vao = 0
        self.vbo = 0
        self.ebo = 0

        self.index_count = 0

        #
        # Kamera
        #

        self.camera_target = np.array(
            [0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        self.camera_distance = 100.0

        self.camera_yaw = 45.0
        self.camera_pitch = 30.0

        self.last_mouse_position = QPoint()

        self.drag_button = None

        self.wireframe = False

    #
    # OpenGL lifecycle
    #

    def initializeGL(self):

        glEnable(
            GL_DEPTH_TEST
        )

        glEnable(
            GL_MULTISAMPLE
        )

        glEnable(
            GL_CULL_FACE
        )

        glCullFace(
            GL_BACK
        )

        glClearColor(
            0.045,
            0.050,
            0.060,
            1.0,
        )

        vertex_source = (
            load_shader_source(
                "mesh.vert"
            )
        )

        fragment_source = (
            load_shader_source(
                "mesh.frag"
            )
        )

        self.shader_program = (
            create_program(
                vertex_source,
                fragment_source,
            )
        )

    def resizeGL(
        self,
        width: int,
        height: int,
    ):

        ratio = (
            self.devicePixelRatioF()
        )

        glViewport(
            0,
            0,
            int(width * ratio),
            int(height * ratio),
        )

    def paintGL(self):

        glClear(
            GL_COLOR_BUFFER_BIT |
            GL_DEPTH_BUFFER_BIT
        )

        if (
            self.mesh is None
            or
            self.index_count == 0
        ):
            return

        glUseProgram(
            self.shader_program
        )

        eye = (
            self.camera_position()
        )

        view = look_at(
            eye,
            self.camera_target,
            np.array(
                [0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
        )

        aspect = (
            max(self.width(), 1) /
            max(self.height(), 1)
        )

        projection = perspective(
            45.0,
            aspect,
            0.01,
            10000.0,
        )

        model = np.eye(
            4,
            dtype=np.float32,
        )

        self._set_matrix(
            "uModel",
            model,
        )

        self._set_matrix(
            "uView",
            view,
        )

        self._set_matrix(
            "uProjection",
            projection,
        )

        location = glGetUniformLocation(
            self.shader_program,
            "uBaseColor",
        )

        glUniform3f(
            location,
            0.22,
            0.68,
            0.42,
        )

        camera_location = (
            glGetUniformLocation(
                self.shader_program,
                "uCameraPosition",
            )
        )

        glUniform3f(
            camera_location,
            float(eye[0]),
            float(eye[1]),
            float(eye[2]),
        )

        if self.wireframe:
            glPolygonMode(
                GL_FRONT_AND_BACK,
                GL_LINE,
            )
        else:
            glPolygonMode(
                GL_FRONT_AND_BACK,
                GL_FILL,
            )

        glBindVertexArray(
            self.vao
        )

        glDrawElements(
            GL_TRIANGLES,
            self.index_count,
            GL_UNSIGNED_INT,
            ctypes.c_void_p(0),
        )

        glBindVertexArray(0)

        glPolygonMode(
            GL_FRONT_AND_BACK,
            GL_FILL,
        )

    #
    # Mesh
    #

    def open_mesh(
        self,
        path: str,
    ):

        mesh = load_mesh(
            path
        )

        self.mesh = mesh

        self.makeCurrent()

        self._upload_mesh(
            mesh
        )

        self.doneCurrent()

        self.focus_mesh()

        self.update()

        return mesh

    def _upload_mesh(
        self,
        mesh,
    ):

        self._destroy_mesh_buffers()

        vertices = np.asarray(
            mesh.vertices,
            dtype=np.float32,
        )

        normals = np.asarray(
            mesh.vertex_normals,
            dtype=np.float32,
        )

        indices = np.asarray(
            mesh.faces,
            dtype=np.uint32,
        )

        #
        # XYZ + normal XYZ
        #

        data = np.concatenate(
            (
                vertices,
                normals,
            ),
            axis=1,
        ).astype(
            np.float32
        )

        self.vao = glGenVertexArrays(
            1
        )

        self.vbo = glGenBuffers(
            1
        )

        self.ebo = glGenBuffers(
            1
        )

        glBindVertexArray(
            self.vao
        )

        glBindBuffer(
            GL_ARRAY_BUFFER,
            self.vbo,
        )

        glBufferData(
            GL_ARRAY_BUFFER,
            data.nbytes,
            data,
            GL_STATIC_DRAW,
        )

        glBindBuffer(
            GL_ELEMENT_ARRAY_BUFFER,
            self.ebo,
        )

        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            indices.nbytes,
            indices,
            GL_STATIC_DRAW,
        )

        stride = (
            6 *
            np.dtype(
                np.float32
            ).itemsize
        )

        #
        # Position
        #

        glEnableVertexAttribArray(
            0
        )

        glVertexAttribPointer(
            0,
            3,
            GL_FLOAT,
            GL_FALSE,
            stride,
            ctypes.c_void_p(0),
        )

        #
        # Normal
        #

        glEnableVertexAttribArray(
            1
        )

        glVertexAttribPointer(
            1,
            3,
            GL_FLOAT,
            GL_FALSE,
            stride,
            ctypes.c_void_p(
                3 * 4
            ),
        )

        glBindVertexArray(
            0
        )

        self.index_count = int(
            indices.size
        )

    def _destroy_mesh_buffers(
        self,
    ):

        if self.ebo:
            glDeleteBuffers(
                1,
                [self.ebo],
            )

        if self.vbo:
            glDeleteBuffers(
                1,
                [self.vbo],
            )

        if self.vao:
            glDeleteVertexArrays(
                1,
                [self.vao],
            )

        self.vao = 0
        self.vbo = 0
        self.ebo = 0

        self.index_count = 0

    #
    # Camera
    #

    def camera_position(
        self,
    ) -> np.ndarray:

        yaw = math.radians(
            self.camera_yaw
        )

        pitch = math.radians(
            self.camera_pitch
        )

        x = (
            self.camera_distance *
            math.cos(pitch) *
            math.cos(yaw)
        )

        y = (
            self.camera_distance *
            math.cos(pitch) *
            math.sin(yaw)
        )

        z = (
            self.camera_distance *
            math.sin(pitch)
        )

        return (
            self.camera_target +
            np.array(
                [x, y, z],
                dtype=np.float32,
            )
        )

    def focus_mesh(
        self,
    ):

        if self.mesh is None:
            return

        center = np.asarray(
            self.mesh.bounding_box.centroid,
            dtype=np.float32,
        )

        extents = np.asarray(
            self.mesh.bounding_box.extents,
            dtype=np.float32,
        )

        size = max(
            float(
                np.max(extents)
            ),
            1.0,
        )

        self.camera_target = center

        self.camera_distance = (
            size * 2.2
        )

        self.camera_yaw = 45.0
        self.camera_pitch = 30.0

        self.update()

    #
    # UI controls
    #

    def set_wireframe(
        self,
        enabled: bool,
    ):

        self.wireframe = enabled
        self.update()

    #
    # Mouse
    #

    def mousePressEvent(
        self,
        event,
    ):

        self.last_mouse_position = (
            event.position().toPoint()
        )

        self.drag_button = (
            event.button()
        )

    def mouseReleaseEvent(
        self,
        event,
    ):

        self.drag_button = None

    def mouseMoveEvent(
        self,
        event,
    ):

        current = (
            event.position().toPoint()
        )

        delta = (
            current -
            self.last_mouse_position
        )

        self.last_mouse_position = (
            current
        )

        if (
            self.drag_button ==
            Qt.LeftButton
        ):

            self.camera_yaw += (
                delta.x() *
                0.45
            )

            self.camera_pitch += (
                delta.y() *
                0.45
            )

            self.camera_pitch = max(
                -89.0,
                min(
                    89.0,
                    self.camera_pitch,
                ),
            )

            self.update()

        elif self.drag_button in (
            Qt.RightButton,
            Qt.MiddleButton,
        ):

            self._pan_camera(
                delta.x(),
                delta.y(),
            )

    def _pan_camera(
        self,
        dx: float,
        dy: float,
    ):

        eye = (
            self.camera_position()
        )

        forward = normalize(
            self.camera_target -
            eye
        )

        world_up = np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )

        right = normalize(
            np.cross(
                forward,
                world_up,
            )
        )

        up = normalize(
            np.cross(
                right,
                forward,
            )
        )

        scale = (
            self.camera_distance *
            0.0015
        )

        self.camera_target -= (
            right *
            dx *
            scale
        )

        self.camera_target += (
            up *
            dy *
            scale
        )

        self.update()

    def wheelEvent(
        self,
        event,
    ):

        steps = (
            event.angleDelta().y() /
            120.0
        )

        self.camera_distance *= (
            0.87 ** steps
        )

        self.camera_distance = max(
            self.camera_distance,
            0.01,
        )

        self.update()

    #
    # Uniform helpers
    #

    def _set_matrix(
        self,
        name: str,
        matrix: np.ndarray,
    ):

        location = (
            glGetUniformLocation(
                self.shader_program,
                name,
            )
        )

        #
        # numpy koristi row-major,
        # pa GL_TRUE radi transpose.
        #

        glUniformMatrix4fv(
            location,
            1,
            GL_TRUE,
            matrix,
        )

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