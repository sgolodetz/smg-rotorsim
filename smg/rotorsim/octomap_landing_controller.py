import numpy as np

from typing import Optional

from smg.navigation import PathNode, PlanningToolkit
from smg.rigging.cameras import SimpleCamera
from smg.rotory.drones import SimulatedDrone


# noinspection SpellCheckingInspection
class OctomapLandingController:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, planning_toolkit: PlanningToolkit):
        """
        TODO

        :param planning_toolkit:    TODO
        """
        self.__goal_y: Optional[float] = None
        self.__planning_toolkit: PlanningToolkit = planning_toolkit

    # SPECIAL METHODS

    def __call__(self, master_cam: SimpleCamera) -> SimulatedDrone.EState:
        """
        TODO

        :param master_cam:  TODO
        :return:            TODO
        """
        if self.__goal_y is None:
            resolution: float = self.__planning_toolkit.get_tree().get_resolution()
            test_vpos: np.ndarray = self.__planning_toolkit.pos_to_vpos(master_cam.p())
            test_node: PathNode = self.__planning_toolkit.pos_to_node(test_vpos)
            while self.__planning_toolkit.node_is_traversable(test_node, use_clearance=True):
                if not self.__planning_toolkit.point_is_in_bounds(test_vpos):
                    self.__goal_y = None
                    return SimulatedDrone.FLYING

                test_vpos[1] += resolution
                test_node = self.__planning_toolkit.pos_to_node(test_vpos)

            self.__goal_y = test_vpos[1] - resolution

        if master_cam.p()[1] < self.__goal_y:
            # FIXME: The linear gain should be passed in to the constructor, not hard-coded like this.
            linear_gain: float = 0.02
            master_cam.move_v(-linear_gain * 0.5)
            return SimulatedDrone.LANDING
        else:
            self.__goal_y = None
            return SimulatedDrone.IDLE
