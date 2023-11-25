import subprocess
import itertools
import csv
import os

class Tournament:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Tournament, cls).__new__(cls)
        return cls._instance

    def __init__(self, players, runs_per_match=1, display_results=False):
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
        result_csv_file = "simulator_results.csv"
        if os.path.exists(result_csv_file):
            os.remove(result_csv_file)
        command = [
            'python3', 'simulator.py', 
            '--player_1', player1, '--player_2', player2, 
            '--autoplay', '--autoplay_runs', str(self.runs_per_match)
        ]
        subprocess.run(command)  # No need to capture output

if __name__ == "__main__":
    players = ['random_agent', "student_agent2", "student_agent3", "student_agent4", "student_agent5"]
    tournament = Tournament(players, display_results=True)
    tournament.run()
