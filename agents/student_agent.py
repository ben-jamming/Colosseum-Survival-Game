# Student agent: Add your own agent here
from math import inf
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time

from .alphabeta import AlphaBeta
from .utils import utility, generate_children, get_move_from_state

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
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
        }


        new_state = AlphaBeta.get_action(
            generate_children,
            utility,
            state,
            max_depth=1
        )

        # print(new_state)

        new_move = get_move_from_state(state, new_state)

        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")


        return new_move
