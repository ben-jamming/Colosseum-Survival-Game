import json
import itertools

from matplotlib import use

def generate_agent_configs():
    max_depths = range(2, 3, 1)
    branching_factors = [0.08]
    simulation_depths = range(50, 301, 250)
    time_limits = [1.5,1.7]
    exploration_constants = [0.5, 1.0, 1.5]
    breadth_limits = range(100, 101, 1)
    dynamic_policies = [False]
    use_full_ordering = [True, False]
    deepening_policies = [True, False]
    agents = []

    # # Generate MCTS agents
    # for depth, sim_depth, time_limit, exploration_constant in itertools.product(max_depths, simulation_depths, time_limits, exploration_constants):
    #     agent_name = f"MCTS_Depth{depth}_Sim{sim_depth}_Time{time_limit}_Expl{exploration_constant}"
    #     agents.append({
    #         "name": agent_name,
    #         "strategy": "MCTS",
    #         "parameters": {
    #             "max_depth": depth,
    #             "simulation_depth": sim_depth,
    #             "time_limit": time_limit,
    #             "exploration_constant": exploration_constant
    #         }
    #     })

    # Generate AlphaBeta agents
    for depth, breadth, time_limit, ordering, b, deepening in itertools.product(max_depths, breadth_limits, time_limits, use_full_ordering, branching_factors, deepening_policies):
        agent_name = f"AB_T_{time_limit}_D_{depth}_B_{breadth}_{'U' if ordering else 'D'}_b_{b}_dp_{deepening}"
        agents.append({
            "name": agent_name,
            "strategy": "AlphaBeta",
            "parameters": {
                "max_depth": depth,
                "breadth_limit": breadth,
                "time_limit": time_limit,
                "use_full_ordering": ordering,
                "b": b,
                "deepening_policy": deepening
            }
        })
    import os
    # Write the configurations to a JSON file
    file_path = f'{os.path.dirname(os.path.realpath(__file__))}/agent_configurations.json'
    with open(file_path, 'w') as file:
        json.dump({"agents": agents}, file, indent=4)

if __name__ == "__main__":
    generate_agent_configs()
