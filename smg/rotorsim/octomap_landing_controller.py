import numpy as np

from typing import Optional

from smg.navigation import PathNode, PlanningToolkit
from smg.rigging.cameras import SimpleCamera
from smg.rotory.drones import SimulatedDrone


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

    def __call__(self, drone_cur: SimpleCamera) -> SimulatedDrone.EState:
        """
        Run an iteration of the landing controller.

        :param drone_cur:   A camera corresponding to the drone's current pose.
        :return:            The state of the drone after this iteration of the controller.
        """
        # If there is no landing currently in progress:
        if self.__goal_y is None:
            # Perform a downwards search from the current position of the drone to find the ground.
            resolution: float = self.__planning_toolkit.get_tree().get_resolution()

            test_vpos: np.ndarray = self.__planning_toolkit.pos_to_vpos(drone_cur.p())
            test_node: PathNode = self.__planning_toolkit.pos_to_node(test_vpos)

            # Step downwards a voxel at a time until the node being tested is non-traversable.
            while self.__planning_toolkit.node_is_traversable(
                test_node, neighbours=PlanningToolkit.neighbours8, use_clearance=True
            ):
                # If we've stepped outside the bounds of the octree, cancel the landing.
                if not self.__planning_toolkit.is_in_bounds(test_vpos):
                    self.__goal_y = None
                    return SimulatedDrone.FLYING

                # Otherwise, keep stepping downwards.
                test_vpos[1] += resolution
                test_node = self.__planning_toolkit.pos_to_node(test_vpos)

            # Once we've found the ground, check that the landing spot is sufficiently flat. If it isn't,
            # cancel the landing.
            for neighbour_node in PlanningToolkit.neighbours8(test_node):
                if self.__planning_toolkit.node_is_free(neighbour_node):
                    self.__goal_y = None
                    return SimulatedDrone.FLYING

            # If we get this far, set the goal height to that of the centre of the voxel just above the
            # landing spot. (Note that y points downwards in our coordinate system!)
            self.__goal_y = test_vpos[1] - resolution

        # If the height of the drone is above the goal height, tell the drone to move downwards. If not, the landing
        # has finished.
        if drone_cur.p()[1] < self.__goal_y:
            drone_cur.move_v(-self.__linear_gain * 0.5)
            return SimulatedDrone.LANDING
        else:
            self.__goal_y = None
            return SimulatedDrone.IDLE
