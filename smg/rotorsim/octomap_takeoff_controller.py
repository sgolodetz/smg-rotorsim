from typing import Optional

from smg.navigation import PlanningToolkit
from smg.rigging.cameras import SimpleCamera
from smg.rotory.drones import Drone


# noinspection SpellCheckingInspection
class OctomapTakeoffController:
    """A takeoff controller for a simulated drone."""

    # CONSTRUCTOR

    def __init__(self, planning_toolkit: PlanningToolkit, *, takeoff_height: float = 1.0, velocity: float = 1.0):
        """
        Construct a takeoff controller for a simulated drone.

        :param planning_toolkit:    The planning toolkit (used for traversability checking).
        :param takeoff_height:      The height (in m) above the ground to which the drone should take off.
        :param velocity:            TODO
        """
        self.__goal_y: Optional[float] = None
        self.__planning_toolkit: PlanningToolkit = planning_toolkit
        self.__takeoff_height: float = takeoff_height
        self.__velocity: float = velocity

    # SPECIAL METHODS

    def __call__(self, drone_cur: SimpleCamera, time_offset: float) -> Drone.EState:
        """
        Run an iteration of the takeoff controller.

        :param drone_cur:   A camera corresponding to the drone's current pose.
        :param time_offset: TODO
        :return:            The state of the drone after this iteration of the controller.
        """
        # If there is no takeoff currently in progress:
        if self.__goal_y is None:
            # Set the goal height based on the current height of the drone (which is on the ground) and the desired
            # takeoff height. (Note that y points downwards in our coordinate system!)
            self.__goal_y = drone_cur.p()[1] - self.__takeoff_height

            # TODO: Choose a goal height that will make sure that the drone doesn't end up in a wall.

        # If the height of the drone is below the goal height, tell the drone to move upwards. If not, the takeoff
        # has finished.
        if drone_cur.p()[1] > self.__goal_y:
            drone_cur.move_v(time_offset * self.__velocity)
            return Drone.TAKING_OFF
        else:
            self.__goal_y = None
            return Drone.FLYING
