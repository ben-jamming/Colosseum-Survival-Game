# Student agent: Add your own agent here
from logging import root
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time
from copy import deepcopy
import math
# import math #MUST ADD THIS TO REQUIREMENTS.TXT!!!
EXP_PARAM = 0.01 # this is most important param to tune
TIME_LIMIT = 1.95 # we will have to decrease this to 1.95
MAX_STEP = 3 # this one doesn't matter, dont tune it, just leave it
DEFAULT_SIMULATIONS = 1
GENERATE_CHILDREN = 20 # the smaller this is , the better the performance for some odd reason
CHILD_GENERATION_DECAY = 0
SIMULATION_DECAY = 50 # decay for worth of simulation results
SIMULATION_CONST = 1

@register_agent("student_agent4")
class StudentAgent4(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent4, self).__init__()
        self.name = "StudentAgent4"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.autoplay = True

        # define Monte Carlo Tree Searc
    class MCTS:
        # define a node of the Monte Carlo Tree
        class Node:
            def __init__(self, board, mypos, advpos, dir = 0, parent = None):
                self.board = board
                # self.move = move
                self.mypos = mypos
                self.advpos = advpos
                self.parent = parent
                self.visits = 0
                self.wins = 0
                self.depth = 0 if parent is None else  parent.depth + 1
                self.dir = dir
                self.children = []
                

            def uct_evaluator(self,node, exp_param):
                if( node.visits == 0 and exp_param == 0): return 0
                if( node.visits == 0 and exp_param != 0): return 1000000000
                return (node.wins / node.visits) + exp_param * math.sqrt(math.log(node.parent.visits)/node.visits)

            def addVisit(self):
               self. visit+=1
            
            def addWin(self, w):
                self.wins+=w

            def getRandomChild(self):
                if not self.children:
                    return None
                return self.children[np.random.randint(0, len(self.children))]
            
            def getMaxChild(self, explore = True):
                if(not self.children):
                    return None
                if explore : return max(self.children, key=lambda x : self.uct_evaluator(node=x, exp_param = EXP_PARAM))
                else: return max(self.children, key=lambda x : self.uct_evaluator(x, exp_param = 0))

        
        def __init__(self, board, mypos, advpos):
            self.root = self.Node(board, mypos, advpos)

    

        

    def check_endgame(self, my_pos, adv_pos, board_size, chess_board):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            # print("RETURNING FALSE")
            return False, p0_score, p1_score
        # player_win = None
        # win_blocks = -1
        # if p0_score > p1_score:
        #     player_win = 0
        #     win_blocks = p0_score
        # elif p0_score < p1_score:
        #     player_win = 1
        #     win_blocks = p1_score
        # else:
        #     player_win = -1  # Tie
        # if player_win >= 0:
        #     logging.info(
        #         f"Game ends! Player {self.player_names[player_win]} wins having control over {win_blocks} blocks!"
        #     )
        # else:
        #     logging.info("Game ends! It is a Tie!")
        return True, p0_score, p1_score

    def random_walk(self, chess_board, my_pos, adv_pos, max_step = MAX_STEP):
        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)
        # steps2 = np.random.randint(0, max_step + 1)
        # steps = max(steps, steps2)

        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir

    def set_barrier(self, chess_board, r, c, dir):
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True



    def expand(self, node:MCTS.Node):
        # print('expanding')
        cur_board = node.board
        mypos = deepcopy(node.mypos)
        advpos = deepcopy(node.advpos)

        valid_moves = self.generate_valid_moves(cur_board, mypos, advpos)
        # print(len(valid_moves) )
        # print('\n----')
        for move in valid_moves:
            
            new_pos , dir = move
            
            new_board = deepcopy(cur_board)
            self.set_barrier(new_board, new_pos[0], new_pos[1], dir)
            # print('\nvalid move')

            child = self.MCTS.Node(board=new_board, parent = node, mypos = new_pos, advpos = advpos, dir = dir)
            # if game_over and p1>p0: # TODO check this optimization, must be a better way to do it
                # self.propagate_to_parent(child, 1) # TODO read this: this is just something im trying out, just want winning positions to be valued more highly, 
                #so I propagate twice if one of the children is a winner -- doesn't make sense, but has some positive impact
            node.children.append(child)
        
        # print('done expanding tree')


#generate a random list of valid moves from a node position of the form ( (r,c), dir )
    def generate_valid_moves(self, chess_board, mypos, advpos, numValidMoves=GENERATE_CHILDREN):
        moves = []
        _ = 0
        while len(moves) == 0 or _ < numValidMoves:
        # for _ in range(0, GENERATE_CHILDREN): #TODO make sure there are valid moves
            my_pos, dir = self.random_walk(deepcopy(chess_board), tuple(mypos), tuple(advpos) )
            if (my_pos, dir) not in moves:
                moves.append((tuple(my_pos), deepcopy(dir) ))
            _ = _ + 1
        # print('generated valid moves')
        return moves


    def select(self, root:MCTS.Node):
        cur = root
        while(cur.children):
            cur = cur.getMaxChild()
        return cur


    def propagate_to_parent(self, node:MCTS.Node, win):
        cur = node
        while(cur is not None):
            cur.visits+=1
            cur.wins+=win
            cur = cur.parent


    def simulate(self,node):
        my_pos = deepcopy(node.mypos)
        adv_pos = deepcopy(node.advpos)
        cur_board = deepcopy(node.board) 

        i=0
        game_over = False
        # print("here is the length")
        # print(len(cur_board))
        # game_over, p0, p1 = self.check_endgame(my_pos, adv_pos, len(cur_board), cur_board)
        # print("GAME OVER IS " + str(game_over))
        # if game_over: 
        #         print(p0>p1)
        #         if p0 > p1 : return 1 
        #         else: return 0

        while(True):
            # print("finding a move \n")
            if(i%2==1): # if it is my turn
                # print("start random walk my turn")
                my_pos, dir = self.random_walk( cur_board ,tuple(my_pos), tuple(adv_pos))
                # print('end random walk')
                self.set_barrier(cur_board, my_pos[0], my_pos[1], dir)
            else:
                # print("start random walk - adv turn")
                adv_pos, dir = self.random_walk( cur_board,tuple(adv_pos), tuple(my_pos) )
                # print('end random walk')
                self.set_barrier(cur_board, adv_pos[0], adv_pos[1], dir)

            game_over, p0, p1 = self.check_endgame(my_pos, adv_pos, len(cur_board), cur_board)
            # print("game is not won yet - simuatlion")
            if game_over: 
                # print("simulation complete")
                decay = SIMULATION_CONST * math.exp(-(node.depth + i/2)/SIMULATION_DECAY)
                if p0 > p1 : return decay
                if p1 > p0: return -decay
                else: return 0
            i+=1
        


        
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
        # dummy return
        global MAX_STEP 
        MAX_STEP = max_step
        tree = self.MCTS(board = chess_board,mypos= my_pos, advpos=adv_pos)
        limit_time = time.time() + TIME_LIMIT

        numIter = 0
        while time.time() < limit_time:
            best_node = self.select(tree.root) # select best leaf node

            if best_node is None:
                best_node = tree.root
            
            game_over, _ , _ = self.check_endgame(best_node.mypos, best_node.advpos,len(best_node.board), best_node.board )

            if(not game_over):
                self.expand(best_node)
            else:
                decay = SIMULATION_CONST * math.exp(-best_node.depth/SIMULATION_DECAY)
                if p0 > p1 : 
                    self.propagate_to_parent(best_node, decay)
                if p0 < p1 : 
                    self.propagate_to_parent(best_node, -decay)
                if p0 == p1 : 
                    self.propagate_to_parent(best_node, 0)
                continue #TODO how to treat if it is not the child of the root

            explorationNode = best_node.getRandomChild()
            # print("got random child")

            #TODO : 
            if not explorationNode:
                continue

    
                # print("null so set to best node")
            
            game_over, p0, p1 = self.check_endgame(explorationNode.mypos, explorationNode.advpos, len(explorationNode.board), explorationNode.board)

            if(game_over): 
                decay = SIMULATION_CONST * math.exp(-explorationNode.depth/SIMULATION_DECAY)
                if p0 > p1 : 
                    self.propagate_to_parent(explorationNode, decay)
                if p0 < p1 : 
                    self.propagate_to_parent(explorationNode, -decay)
                if p0 == p1 : 
                    self.propagate_to_parent(explorationNode, 0)
                continue #TODO how to treat if it is not the child of the root

            win = self.simulate(explorationNode)
            self.propagate_to_parent(explorationNode, win)
            numIter += 1

        print("NumIter = " + str(numIter) + "\n")
        #     print('end of loop\n\n\n')
        for m in tree.root.children:
            game_over, p0, p1 = self.check_endgame(m.mypos, m.advpos, len(m.board), m.board)
            if p0 > p1 : 
                return m.mypos, m.dir
            if p0 < p1 : 
                tree.root.children.remove(m)

        optimal_node = tree.root.getMaxChild(explore=False)
        # return my_pos, self.dir_map["u"]
        return optimal_node.mypos, optimal_node.dir
