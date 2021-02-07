This is an example of different searches of solutions to problems in different methods, mainly heuristic search and cost based search,
The first Map problem is a search through a given map file at: framework\db\tlv_streets_map
The second is a MDA search problem which is set through these specific rules, they are described in comments at the .py files and are:

The program consists of weighted A*, A* epsilon, anytime A* and uniform cost searching algorithms located in: framework\graph_search
The heuristic functions for both problems and for different cost functions are in problems\mda_heuristics and problems\map_heuristics in accordance.
The search algorithms are usable for every given problem described through a class inheriting from GraphProblemState, given examples are mda_problem and map_problem.

