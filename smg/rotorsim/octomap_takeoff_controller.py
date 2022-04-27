from typing import Optional

from smg.navigation import PlanningToolkit
from smg.rigging.cameras import SimpleCamera
from smg.rotory.drones import SimulatedDrone


# noinspection SpellCheckingInspection
class OctomapTakeoffController:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, planning_toolkit: PlanningToolkit, target_height: float = 1.0):
        self.__goal_y: Optional[float] = None
        self.__planning_toolkit: PlanningToolkit = planning_toolkit
        self.__target_height: float = target_height

    # SPECIAL METHODS

    def __call__(self, master_cam: SimpleCamera) -> SimulatedDrone.EState:
        if self.__goal_y is None:
            self.__goal_y = master_cam.p()[1] - self.__target_height

            # TODO: Choose a height that will make sure that the drone doesn't end up in a wall.

        if master_cam.p()[1] > self.__goal_y:
            # FIXME: The linear gain should be passed in to the constructor, not hard-coded like this.
            linear_gain: float = 0.02
            master_cam.move_v(linear_gain * 0.5)
            return SimulatedDrone.TAKING_OFF
        else:
            self.__goal_y = None
            return SimulatedDrone.FLYING
