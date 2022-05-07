import numpy as np

from typing import Optional

from smg.navigation import PlanningToolkit
from smg.rigging.cameras import SimpleCamera
from smg.rotory.drones import Drone


# noinspection SpellCheckingInspection
class OctomapLandingController:
    """A landing controller for a simulated drone."""

    # CONSTRUCTOR

    def __init__(self, planning_toolkit: PlanningToolkit, *, linear_gain: float):
        """
        Construct a landing controller for a simulated drone.

        :param planning_toolkit:    The planning toolkit (used for traversability checking).
        :param linear_gain:         The amount by which control inputs will be multiplied for linear drone movements.
        """
        self.__goal_y: Optional[float] = None
        self.__linear_gain: float = linear_gain
        self.__planning_toolkit: PlanningToolkit = planning_toolkit

    # SPECIAL METHODS

    def __call__(self, drone_cur: SimpleCamera) -> Drone.EState:
        """
        Run an iteration of the landing controller.

        :param drone_cur:   A camera corresponding to the drone's current pose.
        :return:            The state of the drone after this iteration of the controller.
        """
        # If there is no landing currently in progress:
        if self.__goal_y is None:
            # Try to find a patch of flat ground below the current position of the drone.
            ground_vpos: Optional[np.ndarray] = self.__planning_toolkit.find_flat_ground_below(drone_cur.p())

            # If that succeeds, set the goal height to that of the centre of the voxel just above the ground.
            # (Note that y points downwards in our coordinate system!)
            if ground_vpos is not None:
                self.__goal_y = ground_vpos[1] - self.__planning_toolkit.get_tree().get_resolution()

            # Otherwise, clear the goal height and cancel the landing.
            else:
                self.__goal_y = None
                return Drone.FLYING

        # If the height of the drone is above the goal height, tell the drone to move downwards. If not, the landing
        # has finished.
        if drone_cur.p()[1] < self.__goal_y:
            drone_cur.move_v(-self.__linear_gain * 0.5)
            return Drone.LANDING
        else:
            self.__goal_y = None
            return Drone.IDLE
