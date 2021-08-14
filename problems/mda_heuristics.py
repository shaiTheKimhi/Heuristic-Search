import numpy as np
import networkx as nx
from typing import *

from framework import *
from .mda_problem import *
from .cached_air_distance_calculator import CachedAirDistanceCalculator


__all__ = ['MDAMaxAirDistHeuristic', 'MDASumAirDistHeuristic',
           'MDAMSTAirDistHeuristic', 'MDATestsTravelDistToNearestLabHeuristic']




class MDAMaxAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-Max-AirDist'

    def __init__(self, problem: GraphProblem):
        super(MDAMaxAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This method calculated a lower bound of the distance of the remaining path of the ambulance,
         by calculating the maximum distance within the group of air distances between each two
         junctions in the remaining ambulance path. We don't consider laboratories here because we
         do not know what laboratories would be visited in an optimal solution.
        """

        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        all_certain_junctions_in_remaining_ambulance_path = \
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)
        if len(all_certain_junctions_in_remaining_ambulance_path) < 2:
            return 0

        return max(self.cached_air_distance_calculator.get_air_distance_between_junctions(j1, j2)
                   for j1 in all_certain_junctions_in_remaining_ambulance_path for j2 in all_certain_junctions_in_remaining_ambulance_path)


class MDASumAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-Sum-AirDist'

    def __init__(self, problem: GraphProblem):
        super(MDASumAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic evaluates the distance of the remaining ambulance route in the following way:
        It builds a path that starts in the current ambulance's location, and each next junction in
         the path is the (air-distance) nearest junction (to the previous one in the path) among
         all certain junctions (in `all_certain_junctions_in_remaining_ambulance_path`) that haven't
         been visited yet.
        The remaining distance estimation is the cost of this built path.
        Note that we ignore here the problem constraints (like enforcing the #matoshim and free
         space in the ambulance's fridge). We only make sure to visit all certain junctions in
         `all_certain_junctions_in_remaining_ambulance_path`.

        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        ##TODO: search for code optimization, could be usage of iterative methods for this calculation

        all_certain_junctions_in_remaining_ambulance_path = \
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)

        if len(all_certain_junctions_in_remaining_ambulance_path) < 2:
            return 0

        junctions = all_certain_junctions_in_remaining_ambulance_path
        length = len(junctions)
        

        sum_dist = 0
        curr_pos = state.current_site if isinstance(state.current_site, Junction) else state.current_site.location
        for i in range(len(all_certain_junctions_in_remaining_ambulance_path)):
            min_dist = float('inf')
            min_jun = None
            all_certain_junctions_in_remaining_ambulance_path.remove(curr_pos)
            #get closest junction to current junction
            for nxt in all_certain_junctions_in_remaining_ambulance_path:
                dist = self.cached_air_distance_calculator.get_air_distance_between_junctions(curr_pos, nxt)
                if dist < min_dist:
                    min_dist = dist
                    min_jun = nxt
            if min_jun is None:
                return sum_dist
            sum_dist += min_dist
            curr_pos = min_jun

        return sum_dist


class MDAMSTAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-MST-AirDist'

    def __init__(self, problem: GraphProblem):
        super(MDAMSTAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic returns a lower bound for the distance of the remaining route of the ambulance.
        Here this remaining distance is bounded (from below) by the weight of the minimum-spanning-tree
         of the graph, in-which the vertices are the junctions in the remaining ambulance route, and the
         edges weights (edge between each junctions pair) are the air-distances between the junctions.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        return self._calculate_junctions_mst_weight_using_air_distance(
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state))

    def _calculate_junctions_mst_weight_using_air_distance(self, junctions: List[Junction]) -> float:
        g = nx.Graph().to_undirected()
        for j1 in junctions:
            for j2 in junctions:
                if j1 != j2:
                    g.add_edge(j1, j2, weight=self.cached_air_distance_calculator.get_air_distance_between_junctions(j1, j2))
        return nx.minimum_spanning_tree(g).size(weight='weight')



class MDATestsTravelDistToNearestLabHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-TimeObjectiveSumOfMinAirDistFromLab'

    def __init__(self, problem: GraphProblem):
        super(MDATestsTravelDistToNearestLabHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.TestsTravelDistance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic returns a lower bound to the remained tests-travel-distance of the remained ambulance path.
        The main observation is that driving from a laboratory to a reported-apartment does not increase the
         tests-travel-distance cost. So the best case (lowest cost) is when we go to the closest laboratory right
         after visiting any reported-apartment.
        If the ambulance currently stores tests, this total remained cost includes the #tests_on_ambulance times
         the distance from the current ambulance location to the closest lab.
        The rest part of the total remained cost includes the distance between each non-visited reported-apartment
         and the closest lab (to this apartment) times the roommates in this apartment (as we take tests for all
         roommates).
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        problem = self.problem

        def air_dist_to_closest_lab(junction: Junction) -> float:
            """
            Returns the distance between `junction` and the laboratory that is closest to `junction`.
            """
            return min(self.cached_air_distance_calculator.get_air_distance_between_junctions(junction, lab.location) for lab in problem.problem_input.laboratories)
            # We return air distance, probably good but might need to replace with real distance (not probable)

        #taken_tests = state.get_total_nr_tests_taken_and_stored_on_ambulance()
        return sum(ap.nr_roommates * air_dist_to_closest_lab(ap.location) for ap in problem.get_reported_apartments_waiting_to_visit(state))\
            + state.get_total_nr_tests_taken_and_stored_on_ambulance() * air_dist_to_closest_lab(state.current_location)

