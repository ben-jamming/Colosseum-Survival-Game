import json
from .student_agent import StudentAgent
from store import register_agent

def create_and_register_agent(config):
    def agent_factory():
        return StudentAgent(name=config['name'], strategy=config['strategy'], **config['parameters'])
    register_agent(config['name'])(agent_factory)


def load_agents_from_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
        agents = [create_and_register_agent(agent_config) for agent_config in config['agents']]
        return agents
