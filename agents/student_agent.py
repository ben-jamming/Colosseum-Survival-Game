# Student agent: Add your own agent here
from agents.agent import Agent
from agents.mcts import MCTS
from store import register_agent
import numpy as np
import time
from .bitboard import BitBoard


from .alphabeta import AlphaBeta
from .utils import utility, generate_children

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self, name="StudentAgent", strategy="MCTS", dynamic_policy=True, **kwargs):
        super(StudentAgent, self).__init__()
        self.name = name
        self.dynamic_policy = dynamic_policy
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
        board_number = BitBoard(chess_board)
        state = {
            'board': board_number,
            'player': my_pos,
            'adversary': adv_pos,
            'max_step': max_step,
            'is_player_turn': True,
            'action_history': [],
        }

        start_time = time.time()

        wall_count = np.sum(chess_board[:,:,3])

        progression = wall_count / (len(chess_board) * len(chess_board[0]))

        if self.dynamic_policy:
            print("PROGRESSION: ", progression)
            if progression < 0.15 or progression > 0.4:
                print("USING AlphaBeta")
                self.strategy = "AlphaBeta"
            else:
                print("USING MCTS")
                self.strategy = "MCTS"

        if self.strategy == "MCTS":
            new_action = MCTS.get_next_move(
                generate_children,
                utility,
                state,
                max_depth=self.kwargs.get('max_depth',2),
                simulation_depth=self.kwargs.get('simulation_depth',300),
                time_limit=self.kwargs.get('time_limit',1.0),
                memory_limit=500,
                iterations=float('inf'),
                exploration_constant=self.kwargs.get('exploration_constant',0.5)
            )
        elif self.strategy == "AlphaBeta":
            # iterative deepening
            start_time = time.time()
            max_depth = self.kwargs.get('max_depth',2)

            if not self.kwargs.get('deepening', False):
                new_action = AlphaBeta.get_action(
                    generate_children,
                    utility,
                    state,
                    max_depth=max_depth,
                    time_limit=self.kwargs.get('time_limit',1.0),
                    breadth_limit=self.kwargs.get('breadth_limit',400),
                    start_ab=self.kwargs.get('start_ab', (float('-inf'), float('inf')))
                )
            else:
              depth = 1
              print("DOING DEEPENING")
              while time.time() - start_time < 0.5 and depth <= max_depth:
                  new_action = AlphaBeta.get_action(
                      generate_children,
                      utility,
                      state,
                      max_depth=depth,
                      time_limit=self.kwargs.get('time_limit',1.0),
                      breadth_limit=self.kwargs.get('breadth_limit',400),
                      start_ab=self.kwargs.get('start_ab', (float('-inf'), float('inf')))
                  )
                  depth += 1

        elif self.strategy == "Random":
            new_action = generate_children(state)[np.random.randint(len(generate_children(state)))]
            
        else:
            raise ValueError("Invalid strategy")


        time_taken = time.time() - start_time
        
        print(f"{ self.name} turn took ", time_taken, "seconds.")
        # print name
        # print chosen action
        # print("My MCTS AI's action: ", new_action)
        # print walls at that cell
        # print("Walls at that cell: ", chess_board[new_action[0][0], new_action[0][1], 0:4])


        return new_action
