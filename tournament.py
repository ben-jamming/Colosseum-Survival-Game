import subprocess
import itertools
import csv
import re

class Tournament:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Tournament, cls).__new__(cls)
        return cls._instance

    def __init__(self, players, runs_per_match=10, display_results=False):
        self.players = players
        self.runs_per_match = runs_per_match
        self.display_results = display_results
        self.game_results = []

    def run(self):
        """Runs the tournament."""
        for player1, player2 in itertools.combinations(self.players, 2):
            output = self._run_games(player1, player2)
            player_a_win_percentage, player_b_win_percentage, player_a_max_time, player_b_max_time = self._parse_results(output, player1, player2)
            self.game_results.append({
                'player_a': player1,
                'player_b': player2,
                'player_a_win_percentage': player_a_win_percentage,
                'player_b_win_percentage': player_b_win_percentage,
                'player_a_max_time': player_a_max_time,
                'player_b_max_time': player_b_max_time
            })
        self._write_to_csv()


    def _run_games(self, player1, player2):
        """Runs a set of games between two players and returns the results."""
        command = [
            'python3', 'simulator.py', 
            '--player_1', player1, '--player_2', player2, 
            '--autoplay', '--autoplay_runs', str(self.runs_per_match)
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        print("Output from simulator.py:")
        print(result.stdout)
        if result.stderr:
            print("Errors from simulator.py:")
            print(result.stderr)
        return result.stdout  # Ensure this returns a string


    def _parse_results(self, output, player1, player2):
        """Parses the output to extract win percentages and max turn times."""
        player_a_win_percentage = 0.0
        player_b_win_percentage = 0.0
        player_a_max_time = 0.0
        player_b_max_time = 0.0

        win_percentage_pattern = r"Player [AB] win percentage: ([0-9.]+)"
        max_time_pattern = r"Maxium turn time was ([0-9.]+) seconds"

        for line in output.split('\n'):
            win_percentage_match = re.search(win_percentage_pattern, line)
            max_time_match = re.search(max_time_pattern, line)

            if win_percentage_match:
                if "Player A" in win_percentage_match.group():
                    player_a_win_percentage = float(win_percentage_match.group(1))
                elif "Player B" in win_percentage_match.group():
                    player_b_win_percentage = float(win_percentage_match.group(1))

            if max_time_match:
                if player_a_max_time < float(max_time_match.group(1)):
                    player_a_max_time = float(max_time_match.group(1))
                elif player_b_max_time < float(max_time_match.group(1)):
                    player_b_max_time = float(max_time_match.group(1))

        return player_a_win_percentage, player_b_win_percentage, player_a_max_time, player_b_max_time

    def _write_to_csv(self, filename="tournament_results.csv"):
        """Writes the game results to a CSV file."""
        if not self.game_results:
            raise ValueError("No game results to write to CSV.")

        keys = self.game_results[0].keys()
        with open(filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.game_results)
        print(f"Tournament results written to {filename}")

if __name__ == "__main__":
    players = ['student_agent', 'random_agent']
    tournament = Tournament(players, display_results=True)
    tournament.run()
