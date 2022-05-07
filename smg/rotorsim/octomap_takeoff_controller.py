from typing import Optional

from smg.navigation import PlanningToolkit
from smg.rigging.cameras import SimpleCamera
from smg.rotory.drones import Drone


# noinspection SpellCheckingInspection
class OctomapTakeoffController:
    """A takeoff controller for a simulated drone."""

    # CONSTRUCTOR

    def __init__(self, planning_toolkit: PlanningToolkit, *, linear_gain: float, height_offset: float = 1.0):
        """
        Construct a takeoff controller for a simulated drone.

        :param planning_toolkit:    The planning toolkit (used for traversability checking).
        :param linear_gain:         The amount by which control inputs will be multiplied for linear drone movements.
        :param height_offset:       The height offset (in m) above the ground to which the drone should take off.
        """
        self.__goal_y: Optional[float] = None
        self.__height_offset: float = height_offset
        self.__linear_gain: float = linear_gain
        self.__planning_toolkit: PlanningToolkit = planning_toolkit

    # SPECIAL METHODS

    def __call__(self, drone_cur: SimpleCamera) -> Drone.EState:
        """
        Run an iteration of the takeoff controller.

        :param drone_cur:   A camera corresponding to the drone's current pose.
        :return:            The state of the drone after this iteration of the controller.
        """
        # If there is no takeoff currently in progress:
        if self.__goal_y is None:
            # Set the goal height based on the current height of the drone (which is on the ground) and the desired
            # height offset above the ground. (Note that y points downwards in our coordinate system!)
            self.__goal_y = drone_cur.p()[1] - self.__height_offset

            # TODO: Choose a goal height that will make sure that the drone doesn't end up in a wall.

        # If the height of the drone is below the goal height, tell the drone to move upwards. If not, the takeoff
        # has finished.
        if drone_cur.p()[1] > self.__goal_y:
            drone_cur.move_v(self.__linear_gain * 0.5)
            return Drone.TAKING_OFF
        else:
            self.__goal_y = None
            return Drone.FLYING
