import numpy as np

from OpenGL.GL import *
from typing import Callable, Generic, List, Optional, Tuple, TypeVar

from smg.opengl.opengl_framebuffer import OpenGLFrameBuffer
from smg.opengl.opengl_matrix_context import OpenGLMatrixContext
from smg.opengl.opengl_util import OpenGLUtil
from smg.rigging.helpers import CameraPoseConverter


# TYPE VARIABLE

Scene = TypeVar('Scene')


# MAIN CLASS

class SceneRenderer(Generic[Scene]):
    """Used to render the scene for the drone simulator."""

    # CONSTRUCTOR

    def __init__(self):
        """Construct a scene renderer."""
        self.__framebuffer: Optional[OpenGLFrameBuffer] = None

    # DESTRUCTOR

    def __del__(self):
        """Destroy the renderer."""
        self.terminate()

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the renderer's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Destroy the renderer at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC STATIC METHODS

    @staticmethod
    def render(render_scene: Callable[[], None], *, light_dirs: Optional[List[np.ndarray]] = None,
               use_backface_culling: bool = False) -> None:
        """
        Render the scene with the specified directional lighting.

        .. note::
            If light_dirs is None, default light directions will be used. For no lights at all, pass in [].

        :param render_scene:            A function that can be called to render the scene itself.
        :param light_dirs:              The directions from which to light the scene (optional).
        :param use_backface_culling:    Whether or not to use back-face culling.
        """
        # If the light directions haven't been explicitly specified, use the defaults.
        if light_dirs is None:
            pos: np.ndarray = np.array([0.0, -2.0, -1.0, 0.0])
            light_dirs = [pos, -pos]

        # Conversely, if too many light directions have been specified, raise an exception.
        elif len(light_dirs) > 8:
            raise RuntimeError("At most 8 light directions can be specified")

        # Save various attributes so that they can be restored later.
        glPushAttrib(GL_DEPTH_BUFFER_BIT | GL_ENABLE_BIT | GL_LIGHTING_BIT)

        # Enable lighting.
        glEnable(GL_LIGHTING)

        # Set up the directional lights.
        for i in range(len(light_dirs)):
            light_idx: int = GL_LIGHT0 + i
            glEnable(light_idx)
            glLightfv(light_idx, GL_DIFFUSE, np.array([1, 1, 1, 1]))
            glLightfv(light_idx, GL_SPECULAR, np.array([1, 1, 1, 1]))
            glLightfv(light_idx, GL_POSITION, light_dirs[i])

        # Enable colour-based materials (i.e. let material properties be defined by glColor).
        glEnable(GL_COLOR_MATERIAL)

        # Enable back-face culling.
        if use_backface_culling:
            glCullFace(GL_BACK)
            glEnable(GL_CULL_FACE)

        # Enable depth testing.
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_DEPTH_TEST)

        # Render the scene itself.
        render_scene()

        # Restore the attributes to their previous states.
        glPopAttrib()

    # PUBLIC METHODS

    def render_to_image(self, render_scene: Callable[[], None], world_from_camera: np.ndarray,
                        image_size: Tuple[int, int], intrinsics: Tuple[float, float, float, float], *,
                        light_dirs: Optional[List[np.ndarray]] = None, use_backface_culling: bool = False) \
            -> np.ndarray:
        """
        Render the scene to an image.

        .. note::
            If light_dirs is None, default light directions will be used. For no lights at all, pass in [].

        :param render_scene:            A function that can be called to render the scene itself.
        :param world_from_camera:       The pose from which to render the scene.
        :param image_size:              The size of image to render, as a (width, height) tuple.
        :param intrinsics:              The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        :param light_dirs:              The directions from which to light the scene (optional).
        :param use_backface_culling:    Whether or not to use back-face culling.
        :return:                        The rendered image.
        """
        # Make sure the OpenGL frame buffer has been constructed and has the right size.
        width, height = image_size
        if self.__framebuffer is None:
            self.__framebuffer = OpenGLFrameBuffer(width, height)
        elif width != self.__framebuffer.width or height != self.__framebuffer.height:
            self.__framebuffer.terminate()
            self.__framebuffer = OpenGLFrameBuffer(width, height)

        with self.__framebuffer:
            # Set the viewport to encompass the whole frame buffer.
            OpenGLUtil.set_viewport((0.0, 0.0), (1.0, 1.0), (width, height))

            # Clear the background to white.
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Set the projection matrix.
            with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
                intrinsics, width, height
            )):
                # Set the model-view matrix.
                with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                    CameraPoseConverter.pose_to_modelview(np.linalg.inv(world_from_camera))
                )):
                    # Render the scene itself with the specified lighting.
                    SceneRenderer.render(render_scene, light_dirs=light_dirs, use_backface_culling=use_backface_culling)

                    # Read the contents of the frame buffer into an image and return it.
                    return OpenGLUtil.read_bgr_image(width, height)

    def terminate(self) -> None:
        """Destroy the renderer."""
        if self.__framebuffer is not None:
            self.__framebuffer.terminate()
            self.__framebuffer = None
