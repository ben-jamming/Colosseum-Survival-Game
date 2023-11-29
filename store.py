AGENT_REGISTRY = {}

# define decorator for registering game agents
# def register_agent(agent_name=""):
#     def decorator(func):
#         if agent_name not in AGENT_REGISTRY:
#             AGENT_REGISTRY[agent_name] = func
#         else:
#             raise AssertionError(
#                 f"Agent {AGENT_REGISTRY[agent_name]} is already registered."
#             )
#         return func

#     return decorator

def register_agent(agent_name):
    def decorator(agent_instance):
        if agent_name not in AGENT_REGISTRY:
            print(f"Registering agent '{agent_name}'")
            AGENT_REGISTRY[agent_name] = agent_instance
        else:
            raise ValueError(f"Agent '{agent_name}' is already registered.")
        return agent_instance
    return decorator