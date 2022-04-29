from typing import Optional

from smg.navigation import PlanningToolkit
from smg.rigging.cameras import SimpleCamera
from smg.rotory.drones import SimulatedDrone


# noinspection SpellCheckingInspection
class OctomapTakeoffController:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, planning_toolkit: PlanningToolkit, *, linear_gain: float, target_height: float = 1.0):
        self.__goal_y: Optional[float] = None
        self.__linear_gain: float = linear_gain
        self.__planning_toolkit: PlanningToolkit = planning_toolkit
        self.__target_height: float = target_height

    # SPECIAL METHODS

    def __call__(self, drone_cur: SimpleCamera) -> SimulatedDrone.EState:
        """
        TODO

        :param drone_cur:   A camera corresponding to the drone's current pose.
        :return:            The state of the drone after this iteration of the controller.
        """
        if self.__goal_y is None:
            self.__goal_y = drone_cur.p()[1] - self.__target_height

            # TODO: Choose a height that will make sure that the drone doesn't end up in a wall.

        if drone_cur.p()[1] > self.__goal_y:
            drone_cur.move_v(self.__linear_gain * 0.5)
            return SimulatedDrone.TAKING_OFF
        else:
            self.__goal_y = None
            return SimulatedDrone.FLYING
