import subprocess
import itertools
import csv
import os
import json
from multiprocessing import Process, Manager, Pool

from numpy import disp
from simulator import Simulator
import argparse


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

    @staticmethod
    def writer_process(queue):
        data_list = []
        try:
            with open("simulation_results.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if csvfile.tell() == 0:  # Write header if file is empty
                    writer.writerow(['p1', 'p2', 
                                    'p1_avg_score', 'p2_avg_score', 
                                    'p1_max_time', 'p2_max_time', 
                                    'p1_wins', 'p2_wins', 'p1_win_pct', 'p2_win_pct', 'board_size'])                         
                while True:
                    result = queue.get()
                    print("Received data from queue:", result)  # New line for debugging
                    if result == "DONE":
                        break
                    data_list.append(result)
                    writer.writerow(result)

            print("All data in the queue:", data_list)
    
        except Exception as e:
            print("Error in writer process:", e)

    def run(self):
        """Runs the tournament."""
        # Create a queue for inter-process communication
        with Manager() as manager:

            queue = manager.Queue()
            writer_proc = Process(target=self.writer_process, args=(queue,))
            writer_proc.start()
            simulations = [(player1, player2, queue) for player1, player2 in itertools.combinations(self.players, 2)]
            

            # IMPORTANT
            # MAKE THIS WHATEVER THE MAXIMUM NUMBER OF THREADS
            # YOU THINK YOU CAN HANDLE IS
            with Pool(1) as pool:
                pool.starmap(self._run_simulation, simulations)

            # Signal the writer process to terminate
            queue.put("DONE")
            writer_proc.join()     

    def _run_simulation(self, player1, player2, queue):
        """Initiates a set of games between two players."""
        args = argparse.Namespace(
            player_1=player1,
            player_2=player2,
            board_size=12, # This is just a default
            board_size_min=6,
            board_size_max=12,
            display=False,
            display_delay=0.4,
            display_save=False,
            display_save_path="plots/",
            autoplay=True,
            autoplay_runs=self.runs_per_match
        )
        simulator = Simulator(args, queue)
        simulator.autoplay()

if __name__ == "__main__":
    with open('agents/agent_configurations.json', 'r') as file:
        config = json.load(file)
        players = [agent['name'] for agent in config['agents']]
        
    tournament = Tournament(players, display_results=True)
    tournament.run()
