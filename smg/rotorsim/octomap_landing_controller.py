import numpy as np

from typing import Optional

from smg.navigation import PlanningToolkit
from smg.rigging.cameras import SimpleCamera
from smg.rotory.drones import Drone


# noinspection SpellCheckingInspection
class OctomapLandingController:
    """A landing controller for a simulated drone."""

    # CONSTRUCTOR

    def __init__(self, planning_toolkit: PlanningToolkit, *, speed: float = 1.0):
        """
        Construct a landing controller for a simulated drone.

        :param planning_toolkit:    The planning toolkit (used for traversability checking).
        :param speed:               The speed (in m/s) at which the drone should descend during the landing.
        """
        self.__goal_y: Optional[float] = None
        self.__planning_toolkit: PlanningToolkit = planning_toolkit
        self.__speed: float = speed

    # SPECIAL METHODS

    def __call__(self, drone_cur: SimpleCamera, time_offset: float) -> Drone.EState:
        """
        Run an iteration of the landing controller.

        :param drone_cur:   A camera corresponding to the drone's current pose.
        :param time_offset: The time offset (in s) since the last iteration of the simulation.
        :return:            The state of the drone after this iteration of the controller.
        """
        # If there is no landing currently in progress:
        if self.__goal_y is None:
            # Try to find a patch of flat ground below the current position of the drone.
            ground_vpos: Optional[np.ndarray] = self.__planning_toolkit.find_flat_ground_below(drone_cur.p())

            # If that succeeds, set the goal height to that of the centre of the voxel just above the ground.
            # (Note that y points downwards in our coordinate system!)
            if ground_vpos is not None:
                self.__goal_y = ground_vpos[1] - self.__planning_toolkit.get_tree_resolution()

            # Otherwise, clear the goal height and cancel the landing.
            else:
                self.__goal_y = None
                return Drone.FLYING

        # If the height of the drone is above the goal height, tell the drone to move downwards. If not, the landing
        # has finished.
        if drone_cur.p()[1] < self.__goal_y:
            drone_cur.move_v(-time_offset * self.__speed)
            return Drone.LANDING
        else:
            self.__goal_y = None
            return Drone.IDLE
