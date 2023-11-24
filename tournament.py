import subprocess
import itertools
import csv

class Tournament:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Tournament, cls).__new__(cls)
        return cls._instance

    def __init__(self, players, runs_per_match=10):
        self.players = players
        self.runs_per_match = runs_per_match
        self.game_results = []

    def run(self):
        """Runs the tournament and writes results to a CSV file."""
        for player1, player2 in itertools.combinations(self.players, 2):
            self.game_results.extend(self._run_games(player1, player2))
        self._write_to_csv()

    def _run_games(self, player1, player2):
        """Runs a set of games between two players and returns the results.

        Args:
            player1 (str): The name of the first player.
            player2 (str): The name of the second player.

        Returns:
            list of dict: A list containing the results of each game.
        """
        command = [
            'python3', 'simulator.py', 
            '--player_1', player1, '--player_2', player2, 
            '--autoplay', '--autoplay_runs', str(self.runs_per_match)
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        print("Simulator Output:", result.stdout)  # Debug print
        print("Simulator Errors:", result.stderr)  # Debug print
        return self._parse_results(result.stdout, player1, player2)

    def _parse_results(self, output, player1, player2):
        """Parses the output from a set of games to extract scores and winners.

        Args:
            output (str): The stdout from the simulator script.
            player1 (str): The name of the first player.
            player2 (str): The name of the second player.

        Returns:
            list of dict: A list containing the parsed game results.
        """
        game_results = []
        for line in output.split('\n'):
            if line.startswith("INFO:Run finished."):
                p1_score, p2_score = [int(s.split(": ")[1]) for s in line.split(", ")]
                winner = player1 if p1_score > p2_score else player2 if p1_score < p2_score else "Tie"
                game_results.append({
                    'Player 1': player1, 'Player 2': player2,
                    'Player 1 Score': p1_score, 'Player 2 Score': p2_score,
                    'Winner': winner
                })
        return game_results

    def _write_to_csv(self, filename="tournament_results.csv"):
        """Writes the game results to a CSV file.

        Args:
            filename (str, optional): The name of the CSV file. Defaults to "tournament_results.csv".
        """
        if not self.game_results:
            raise ValueError("No game results to write to CSV.")

        keys = self.game_results[0].keys()
        with open(filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.game_results)
        print(f"Tournament results written to {filename}")

if __name__ == "__main__":
    players = ['random_agent', 'student_agent']  # Add your agents here
    tournament = Tournament(players)
    tournament.run()
