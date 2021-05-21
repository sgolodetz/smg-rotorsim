import math
import numpy as np
import open3d as o3d
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import threading
import time

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.joysticks import FutabaT6K
from smg.navigation import AStarPathPlanner, OCS_OCCUPIED, PlanningToolkit
from smg.opengl import CameraRenderer, OpenGLImageRenderer, OpenGLMatrixContext, OpenGLTriMesh, OpenGLUtil
from smg.pyoctomap import CM_COLOR_HEIGHT, OctomapUtil, OcTree, OcTreeDrawer
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter, CameraUtil
from smg.rotory.drones import SimulatedDrone
from smg.utility import ImageUtil

from .opengl_scene_renderer import OpenGLSceneRenderer


class DroneSimulator:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, *, drone_mesh_filename: str, intrinsics: Tuple[float, float, float, float],
                 plan_paths: bool = False, planning_octree_filename: Optional[str],
                 scene_mesh_filename: Optional[str], scene_octree_filename: Optional[str],
                 window_size: Tuple[int, int] = (1280, 480)):
        self.__alive: bool = False

        self.__drone: Optional[SimulatedDrone] = None
        self.__drone_mesh: Optional[OpenGLTriMesh] = None
        self.__drone_mesh_filename: str = drone_mesh_filename
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics
        self.__gl_image_renderer: Optional[OpenGLImageRenderer] = None
        self.__gl_scene_renderer: Optional[OpenGLSceneRenderer] = None
        self.__octree_drawer: Optional[OcTreeDrawer] = None
        self.__plan_paths: bool = plan_paths
        self.__planning_octree_filename: Optional[str] = planning_octree_filename
        self.__planning_toolkit: Optional[PlanningToolkit] = None
        self.__should_terminate: threading.Event = threading.Event()
        self.__scene_mesh: Optional[OpenGLTriMesh] = None
        self.__scene_mesh_filename: Optional[str] = scene_mesh_filename
        self.__scene_octree: Optional[OcTree] = None
        self.__scene_octree_filename: Optional[str] = scene_octree_filename
        self.__window_size: Tuple[int, int] = window_size

        # The path planning variables, together with their lock.
        self.__current_pos: Optional[np.ndarray] = None
        self.__interpolated_path: Optional[np.ndarray] = None
        self.__path: Optional[np.ndarray] = None
        self.__path_flags: Optional[np.ndarray] = None
        self.__planning_lock: threading.Lock = threading.Lock()

        # FIXME: These shouldn't be hard-coded.
        voxel_size: float = 0.05
        self.__waypoints: List[np.ndarray] = [
            # np.array([0.5, 0.5, 5.5]) * voxel_size,
            # np.array([-5.5, 10.5, 15.5]) * voxel_size,
            np.array([30.5, 5.5, 5.5]) * voxel_size,
            # np.array([50.5, 0.5, 20.5]) * voxel_size
        ]

        # The threads and conditions.
        self.__planning_thread: Optional[threading.Thread] = None
        self.__planning_is_needed: bool = False
        self.__planning_needed: threading.Condition = threading.Condition(self.__planning_lock)

        self.__alive = True

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the simulator's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the simulator at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def run(self) -> None:
        """Run the drone simulator."""
        # Initialise pygame and some of its modules.
        pygame.init()
        pygame.joystick.init()
        pygame.mixer.init()

        # Make sure pygame always gets the user inputs.
        pygame.event.set_grab(True)

        # Try to determine the joystick index of the Futaba T6K. If no joystick is plugged in, early out.
        joystick_count: int = pygame.joystick.get_count()
        joystick_idx: int = 0
        if joystick_count == 0:
            exit(0)
        elif joystick_count != 1:
            # TODO: Prompt the user for the joystick to use.
            pass

        # Construct and calibrate the Futaba T6K.
        joystick: FutabaT6K = FutabaT6K(joystick_idx)
        joystick.calibrate()

        # Create the window.
        pygame.display.set_mode(self.__window_size, pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Drone Simulator")

        # Construct the OpenGL image renderer.
        self.__gl_image_renderer = OpenGLImageRenderer()

        # Construct the OpenGL scene renderer.
        self.__gl_scene_renderer = OpenGLSceneRenderer()

        # Construct the octree drawer.
        self.__octree_drawer = OcTreeDrawer()
        self.__octree_drawer.set_color_mode(CM_COLOR_HEIGHT)

        # Load in the mesh for the drone, and prepare it for rendering.
        self.__drone_mesh = DroneSimulator.__convert_trimesh_to_opengl(
            DroneSimulator.__load_tello_mesh(self.__drone_mesh_filename)
        )

        # Load in any mesh that has been provided for the scene.
        if self.__scene_mesh_filename is not None:
            self.__scene_mesh = DroneSimulator.__convert_trimesh_to_opengl(
                o3d.io.read_triangle_mesh(self.__scene_mesh_filename)
            )

        # Load in any octree that has been provided for the scene.
        if self.__scene_octree_filename is not None:
            scene_voxel_size: float = 0.05
            self.__scene_octree = OcTree(scene_voxel_size)
            self.__scene_octree.read_binary(self.__scene_octree_filename)

        # Load in the "drone flying" sound.
        pygame.mixer.music.load("C:/smglib/sounds/drone_flying.mp3")

        # Construct the simulated drone.
        width, height = self.__window_size
        self.__drone = SimulatedDrone(
            image_renderer=self.__render_drone_image, image_size=(width // 2, height), intrinsics=self.__intrinsics
        )

        # Construct the camera controller.
        camera_controller: KeyboardCameraController = KeyboardCameraController(
            CameraUtil.make_default_camera(), canonical_angular_speed=0.05, canonical_linear_speed=0.025
        )

        # If we're planning paths, start the path planning thread.
        if self.__plan_paths and self.__planning_octree_filename is not None:
            self.__planning_thread = threading.Thread(target=self.__run_planning)
            self.__planning_thread.start()

        # Prevent the drone's gimbal from being moved until we're ready.
        can_move_gimbal: bool = False

        # Until the simulator should terminate:
        while not self.__should_terminate.is_set():
            # Process any PyGame events.
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    # If Button 0 on the Futaba T6K is set to its "pressed" state:
                    if event.button == 0:
                        # Start playing the "drone flying" sound.
                        if self.__drone.get_state() == SimulatedDrone.IDLE:
                            pygame.mixer.music.play(loops=-1)

                        # Take off.
                        self.__drone.takeoff()
                elif event.type == pygame.JOYBUTTONUP:
                    # If Button 0 on the Futaba T6K is set to its "released" state, land.
                    if event.button == 0:
                        self.__drone.land()
                elif event.type == pygame.QUIT:
                    # If the user wants us to quit, do so.
                    return

            # Quit if both Button 0 and Button 1 on the Futaba T6K are set to their "released" state.
            if joystick.get_button(0) == 0 and joystick.get_button(1) == 0:
                return

            # If the drone is in the idle state, stop the "drone flying" sound.
            if self.__drone.get_state() == SimulatedDrone.IDLE:
                pygame.mixer.music.stop()

            # Update the movement of the drone based on the pitch, roll and yaw values output by the Futaba T6K.
            self.__drone.move_forward(joystick.get_pitch())
            self.__drone.turn(joystick.get_yaw())

            if joystick.get_button(1) == 0:
                self.__drone.move_right(0)
                self.__drone.move_up(joystick.get_roll())
            else:
                self.__drone.move_right(joystick.get_roll())
                self.__drone.move_up(0)

            # If the throttle goes above half-way, enable movement of the drone's gimbal from now on.
            throttle: float = joystick.get_throttle()
            if throttle >= 0.5:
                can_move_gimbal = True

            # If the drone's gimbal can be moved, update its pitch based on the current value of the throttle.
            # Note that the throttle value is in [0,1], so we rescale it to a value in [-1,1] as a first step.
            if can_move_gimbal:
                self.__drone.update_gimbal_pitch(2 * (joystick.get_throttle() - 0.5))

            # Get the drone's image and poses.
            drone_image, drone_camera_w_t_c, drone_chassis_w_t_c = self.__drone.get_image_and_poses()

            # If a path has been planned, draw it.
            acquired: bool = self.__planning_lock.acquire(blocking=False)
            if acquired:
                try:
                    if self.__drone.get_state() == SimulatedDrone.FLYING:
                        self.__current_pos = drone_camera_w_t_c[0:3, 3]
                        self.__planning_is_needed = True
                        self.__planning_needed.notify()
                finally:
                    self.__planning_lock.release()

            # Allow the user to control the free-view camera.
            camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

            # Render the contents of the window.
            self.__render_window(
                drone_chassis_w_t_c=drone_chassis_w_t_c,
                drone_image=drone_image,
                viewing_pose=camera_controller.get_pose()
            )

    def terminate(self) -> None:
        """Destroy the simulator."""
        if self.__alive:
            # Set the termination flag if it isn't set already.
            if not self.__should_terminate.is_set():
                self.__should_terminate.set()

            # Join any running threads.
            if self.__planning_thread is not None:
                self.__planning_thread.join()

            # If the simulated drone exists, destroy it.
            if self.__drone is not None:
                self.__drone.terminate()

            # If the OpenGL scene renderer exists, destroy it.
            if self.__gl_scene_renderer is not None:
                self.__gl_scene_renderer.terminate()

            # If the OpenGL image renderer exists, destroy it.
            if self.__gl_image_renderer is not None:
                self.__gl_image_renderer.terminate()

            # Shut down pygame.
            pygame.quit()

            self.__alive = False

    # PRIVATE METHODS

    def __render_drone_image(self, world_from_camera: np.ndarray, image_size: Tuple[int, int],
                             intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
        """
        TODO

        :param world_from_camera:   TODO
        :param image_size:          TODO
        :param intrinsics:          TODO
        :return:                    TODO
        """
        if self.__scene_mesh is not None:
            return self.__gl_scene_renderer.render_to_image(
                self.__scene_mesh.render, world_from_camera, image_size, intrinsics
            )
        elif self.__scene_octree is not None:
            return self.__gl_scene_renderer.render_to_image(
                lambda: OctomapUtil.draw_octree(self.__scene_octree, self.__octree_drawer),
                world_from_camera, image_size, intrinsics
            )
        else:
            width, height = image_size
            return np.zeros((height, width, 3), dtype=np.uint8)

    def __render_window(self, *, drone_chassis_w_t_c: np.ndarray, drone_image: np.ndarray, viewing_pose: np.ndarray) \
            -> None:
        """Render the contents of the window."""
        # Clear the window.
        OpenGLUtil.set_viewport((0.0, 0.0), (1.0, 1.0), self.__window_size)
        glClearColor(1.0, 1.0, 1.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Render the whole scene from the viewing pose.
        OpenGLUtil.set_viewport((0.0, 0.0), (0.5, 1.0), self.__window_size)

        glPushAttrib(GL_DEPTH_BUFFER_BIT)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_DEPTH_TEST)

        width, height = self.__window_size
        with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
            self.__intrinsics, width // 2, height
        )):
            with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                CameraPoseConverter.pose_to_modelview(viewing_pose)
            )):
                # Render a voxel grid.
                glColor3f(0.0, 0.0, 0.0)
                OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

                # Render coordinate axes.
                CameraRenderer.render_camera(CameraUtil.make_default_camera())

                # Render the scene itself.
                if self.__scene_octree is not None:
                    OpenGLSceneRenderer.render(
                        lambda: OctomapUtil.draw_octree(self.__scene_octree, self.__octree_drawer)
                    )
                elif self.__scene_mesh is not None:
                    OpenGLSceneRenderer.render(self.__scene_mesh.render)

                # Render the mesh for the drone (at its current pose).
                with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(drone_chassis_w_t_c)):
                    OpenGLSceneRenderer.render(lambda: self.__drone_mesh.render())

                # If a path has been planned, draw it.
                acquired: bool = self.__planning_lock.acquire(blocking=False)
                if acquired:
                    try:
                        if self.__path is not None:
                            OpenGLUtil.render_path(
                                self.__path, start_colour=(0, 1, 1), end_colour=(0, 1, 1), width=5,
                                waypoint_colourer=self.__planning_toolkit.occupancy_colourer()
                            )
                            # OpenGLUtil.render_path(
                            #     self.__interpolated_path, start_colour=(1, 1, 0), end_colour=(1, 0, 1), width=5,
                            #     waypoint_colourer=None
                            # )
                    finally:
                        self.__planning_lock.release()

        glPopAttrib()

        # Render the drone image.
        OpenGLUtil.set_viewport((0.5, 0.0), (1.0, 1.0), self.__window_size)
        self.__gl_image_renderer.render_image(ImageUtil.flip_channels(drone_image))

        # Swap the front and back buffers.
        pygame.display.flip()

    def __run_planning(self) -> None:
        """Run the path planning thread."""
        # Load the planning octree.
        voxel_size: float = 0.1
        octree: OcTree = OcTree(voxel_size)
        octree.read_binary(self.__planning_octree_filename)

        # Construct the planning toolkit.
        self.__planning_toolkit = PlanningToolkit(
            octree,
            neighbours=PlanningToolkit.neighbours6,
            node_is_free=lambda n: self.__planning_toolkit.occupancy_status(n) != OCS_OCCUPIED
        )

        # Construct the path planner.
        planner: AStarPathPlanner = AStarPathPlanner(self.__planning_toolkit, debug=False)

        # TODO
        while not self.__should_terminate.is_set():
            with self.__planning_lock:
                while not self.__planning_is_needed:
                    self.__planning_needed.wait(0.1)
                    if self.__should_terminate.is_set():
                        return

                current_pos: Optional[np.ndarray] = self.__current_pos.copy()
                flags: Optional[np.ndarray] = self.__path_flags
                interpolated_path: Optional[np.ndarray] = self.__interpolated_path
                path: Optional[np.ndarray] = self.__path
                waypoints: List[np.ndarray] = self.__waypoints

            if path is None:
                # Plan an initial path through the waypoints.
                start = timer()
                ay: float = 10
                path, flags = planner.plan_multipath(
                    [current_pos] + waypoints,
                    d=PlanningToolkit.l1_distance(ay=ay),
                    h=PlanningToolkit.l1_distance(ay=ay),
                    allow_shortcuts=True,
                    pull_strings=True,
                    use_clearance=True
                )
                end = timer()
                # print(f"Path Planning: {end - start}s")
            elif len(path) > 1:
                path, flags = planner.update_path(current_pos, path, flags)

            # Smooth any path found.
            if path is not None:
                start = timer()
                interpolated_path = PlanningToolkit.interpolate_path(path)
                end = timer()
                # print(f"Path Smoothing: {end - start}s")

            with self.__planning_lock:
                self.__interpolated_path = interpolated_path
                self.__path = path
                self.__path_flags = flags
                self.__planning_is_needed = False

            # TODO
            time.sleep(0.01)

    # PRIVATE STATIC METHODS

    @staticmethod
    def __convert_trimesh_to_opengl(o3d_mesh: o3d.geometry.TriangleMesh) -> OpenGLTriMesh:
        """
        Convert an Open3D triangle mesh to an OpenGL one.

        :param o3d_mesh:    The Open3D triangle mesh.
        :return:            The OpenGL mesh.
        """
        # FIXME: This should probably be moved somewhere more central at some point.
        o3d_mesh.compute_vertex_normals(True)
        return OpenGLTriMesh(
            np.asarray(o3d_mesh.vertices),
            np.asarray(o3d_mesh.vertex_colors),
            np.asarray(o3d_mesh.triangles),
            vertex_normals=np.asarray(o3d_mesh.vertex_normals)
        )

    # noinspection PyArgumentList
    @staticmethod
    def __load_tello_mesh(filename: str) -> o3d.geometry.TriangleMesh:
        """
        Load the DJI Tello mesh from the specified file.

        .. note::
            There is a special function for this because the mesh needs some processing to get it into a usable format.

        :param filename:    The name of the file containing the DJI Tello mesh.
        :return:            The DJI Tello mesh.
        """
        mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(filename)
        mesh.translate(-mesh.get_center())
        mesh.scale(0.002, np.zeros(3))
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([math.pi, 0, 0])))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.array([0, 1, 1]))
        return mesh
