from .agent import Agent
from .random_agent import RandomAgent
from .human_agent import HumanAgent
from .student_agent import StudentAgent
# from .student_agent1 import AlphaAgent
# from .student_agent2 import StudentAgent2
# from .student_agent3 import StudentAgent3
# from .student_agent4 import StudentAgent4
# from .student_agent5 import StudentAgent5
from .agent_factory import load_agents_from_config

import os
config_path = f'{os.path.dirname(os.path.realpath(__file__))}/agent_configurations.json'
load_agents_from_config(config_path)