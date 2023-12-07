import json
import itertools

from matplotlib import use

def generate_agent_configs():
    max_depths = range(2, 3, 1)
    simulation_depths = range(50, 200, 50)
    time_limits = [1.0]
    exploration_constants = [0.5, 1.0, 1.5]
    breadth_limits = range(350, 651, 300)
    dynamic_policies = [False]
    use_full_ordering = [True, False]
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
    for depth, breadth, time_limit, ordering in itertools.product(max_depths, breadth_limits, time_limits, use_full_ordering):
        agent_name = f"AB_Dpth_{depth}_Brth_{breadth}_dp{ordering}"
        agents.append({
            "name": agent_name,
            "strategy": "AlphaBeta",
            "parameters": {
                "max_depth": depth,
                "breadth_limit": breadth,
                "time_limit": time_limit,
                "use_full_ordering": ordering
            }
        })
    import os
    # Write the configurations to a JSON file
    file_path = f'{os.path.dirname(os.path.realpath(__file__))}/agent_configurations.json'
    with open(file_path, 'w') as file:
        json.dump({"agents": agents}, file, indent=4)

if __name__ == "__main__":
    generate_agent_configs()
