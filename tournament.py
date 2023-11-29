import subprocess
import itertools
import csv
import os
import json

class Tournament:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Tournament, cls).__new__(cls)
        return cls._instance

    def __init__(self, players, runs_per_match=5, display_results=False):
        self.players = players
        self.runs_per_match = runs_per_match
        self.display_results = display_results

    def run(self):
        """Runs the tournament."""
        for player1, player2 in itertools.combinations_with_replacement(self.players, 2):
            self._run_simulation(player1, player2)
        # Optionally, read and process the results from the CSV file

    def _run_simulation(self, player1, player2):
        """Initiates a set of games between two players."""
        # clear the results file ny deleting it
        command = [
            'python3', 'simulator.py', 
            '--player_1', player1, '--player_2', player2, 
            '--autoplay', '--autoplay_runs', str(self.runs_per_match)
        ]
        subprocess.run(command)  # No need to capture output
    


if __name__ == "__main__":
    with open('/Users/Ben/Documents/McGill/U3/COMP 424/Colosseum-Survival-Game/agents/agent_config.json', 'r') as file:
        config = json.load(file)
        players = [agent['name'] for agent in config['agents']]
        
    tournament = Tournament(players, display_results=True)
    tournament.run()
