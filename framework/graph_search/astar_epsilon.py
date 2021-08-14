from .graph_problem_interface import *
from .astar import AStar
from typing import Optional, Callable
import numpy as np
import math


class AStarEpsilon(AStar):
    """
    This class implements the (weighted) A*Epsilon search algorithm.
    A*Epsilon algorithm basically works like the A* algorithm, but with
    another way to choose the next node to expand from the open queue.
    """

    solver_name = 'A*eps'

    def __init__(self,
                 heuristic_function_type: HeuristicFunctionType,
                 within_focal_priority_function: Callable[[SearchNode, GraphProblem, 'AStarEpsilon'], float],
                 heuristic_weight: float = 0.5,
                 max_nr_states_to_expand: Optional[int] = None,
                 focal_epsilon: float = 0.1,
                 max_focal_size: Optional[int] = None):
        # A* is a graph search algorithm. Hence, we use close set.
        super(AStarEpsilon, self).__init__(heuristic_function_type, heuristic_weight,
                                           max_nr_states_to_expand=max_nr_states_to_expand)
        self.focal_epsilon = focal_epsilon
        if focal_epsilon < 0:
            raise ValueError(f'The argument `focal_epsilon` for A*eps should be >= 0; '
                             f'given focal_epsilon={focal_epsilon}.')
        self.within_focal_priority_function = within_focal_priority_function
        self.max_focal_size = max_focal_size

    def _init_solver(self, problem):
        super(AStarEpsilon, self)._init_solver(problem)

    def _extract_next_search_node_to_expand(self, problem: GraphProblem) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
         by focusing on the current FOCAL and choosing the node
         with the best within_focal_priority from it.
        """
        if self.open is None or self.open.is_empty():
            return None
        lowest = self.open.pop_next_node()
        lim = (1 + self.focal_epsilon) * lowest.expanding_priority
        focal = [lowest]
        max_size = self.max_focal_size is not None
        while self.open.is_empty() is not True:
            node = self.open.peek_next_node()
            #open is sorted, so if current node expanding priority is larger than limit, we can finish the search
            if node.expanding_priority > lim:
                break
            focal.append(self.open.pop_next_node())
            #checking if we use max size and whether our focal is at max size
            if max_size and self.max_focal_size == len(focal):
                break
        #create heuristic for every node in focal and choose minimum
        secondary_heuristic = [self.within_focal_priority_function(node, problem, self) for node in focal]
        min_index = np.argmin(secondary_heuristic)
        #return all not selected to open and selected will be added to open
        for i in range(len(focal)):
            if i != min_index:
                self.open.push_node(focal[i])

        if self.use_close:
            self.close.add_node(focal[min_index])
        return focal[min_index]
