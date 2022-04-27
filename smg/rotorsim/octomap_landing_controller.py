import numpy as np

from typing import Optional

from smg.navigation import PathNode, PlanningToolkit
from smg.rigging.cameras import SimpleCamera
from smg.rotory.drones import SimulatedDrone


# noinspection SpellCheckingInspection
class OctomapLandingController:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, planning_toolkit: PlanningToolkit, *, linear_gain: float):
        """
        TODO

        :param linear_gain:         TODO
        :param planning_toolkit:    TODO
        """
        self.__goal_y: Optional[float] = None
        self.__linear_gain: float = linear_gain
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
            while self.__planning_toolkit.node_is_traversable(
                test_node, neighbours=PlanningToolkit.neighbours8, use_clearance=True
            ):
                if not self.__planning_toolkit.point_is_in_bounds(test_vpos):
                    self.__goal_y = None
                    return SimulatedDrone.FLYING

                test_vpos[1] += resolution
                test_node = self.__planning_toolkit.pos_to_node(test_vpos)

            for neighbour_node in PlanningToolkit.neighbours8(test_node):
                if self.__planning_toolkit.node_is_free(neighbour_node):
                    self.__goal_y = None
                    return SimulatedDrone.FLYING

            self.__goal_y = test_vpos[1] - resolution

        if master_cam.p()[1] < self.__goal_y:
            master_cam.move_v(-self.__linear_gain * 0.5)
            return SimulatedDrone.LANDING
        else:
            self.__goal_y = None
            return SimulatedDrone.IDLE
