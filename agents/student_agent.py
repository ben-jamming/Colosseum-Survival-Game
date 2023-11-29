# Student agent: Add your own agent here
from math import e, inf
from agents.agent import Agent
from agents.mcts import MCTS
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time

from .alphabeta import AlphaBeta
from .utils import utility, generate_children

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self, name="StudentAgent", strategy="MCTS", **kwargs):
        super(StudentAgent, self).__init__()
        self.name = name
        self.strategy = strategy
        self.kwargs = kwargs
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        state = {
            'board': chess_board,
            'player': my_pos,
            'adversary': adv_pos,
            'max_step': max_step,
            'is_player_turn': True,
            'action_history': [],
        }

        start_time = time.time()

        if self.strategy == "MCTS":
            new_action = MCTS.get_next_move(
                generate_children,
                utility,
                state,
                max_depth=self.kwargs.get('max_depth',2),
                simulation_depth=self.kwargs.get('simulation_depth',100),
                time_limit=self.kwargs.get('time_limit',1.0),
                memory_limit=500,
                iterations=float('inf'),
                exploration_constant=self.kwargs.get('exploration_constant',1.0)
            )
        elif self.strategy == "AlphaBeta":
            new_action = AlphaBeta.get_action(
                generate_children,
                utility,
                state,
                self.kwargs.get('max_depth',2),
                self.kwargs.get('time_limit',1.0),
                self.kwargs.get('breadth_limit',200),
            )
        elif self.strategy == "Random":
            new_action = generate_children(state)[np.random.randint(len(generate_children(state)))]
            
        else:
            raise ValueError("Invalid strategy")


        time_taken = time.time() - start_time
        
        print("My MCTS AI's turn took ", time_taken, "seconds.")


        return new_action
