# Student agent: Add your own agent here
from logging import root
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time
from copy import deepcopy
import math
from .alphabeta import AlphaBeta
from .utils import utility, generate_children

# import math #MUST ADD THIS TO REQUIREMENTS.TXT!!!
EXP_PARAM =0.04 # this is most important param to tune
TIME_LIMIT = 2 # we will have to decrease this to 1.95
MAX_STEP = 3 # this one doesn't matter, dont tune it, just leave it
DEFAULT_SIMULATIONS = 1
GENERATE_CHILDREN = 20 # the smaller this is , the better the performance for some odd reason

@register_agent("alpha_agent")
class AlphaAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(AlphaAgent, self).__init__()
        self.name = "AlphaBetaAgent"
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

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()

        state = {
            'board': chess_board,
            'player': my_pos,
            'adversary': adv_pos,
            'max_step': max_step,
            'is_player_turn': True,
            'action_history': [],
        }

        new_action = AlphaBeta.get_action(
            generate_children,
            utility,
            state,
            max_depth=6,
            time_limit=1.0,
            breadth_limit=200,
        )

        time_taken = time.time() - start_time
        print("My Alpha-Beta AI's turn took ", time_taken, "seconds.")
        return new_action
