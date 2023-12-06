from agents.agent import Agent
from store import register_agent
import numpy as np
import time

# IMPORTANT: WE might have to add these to requirements.txt?
from collections import deque
import heapq
from functools import lru_cache

# NOTE: I defined these at the global scope because I beleive this is fine according to the assignment
# Description. However, we could also just redefine these all within the scope of the alpha-beta
# functions that call them. The reason I didn't do this was because I was unsure about how much overhead
# It would add, especially since we are using lru_cache to cache the results of these functions.
# Whether this would actually be an issue idk.
class BitBoard():

    def __init__(self,npboard=None, n=12):
        self.n = n
        if npboard is not None:
            self.n = npboard.shape[0]
            self.from_array(npboard)

    def bit_index(self, i, j, wall):
        return 4 * (self.n * i + j) + wall

    def from_array(self, npboard):
        n = npboard.shape[0]
        self.board = 0
        self.n = n
        for i in range(n):
            for j in range(n):
                for wall in range(4):
                    if npboard[i, j, wall]:
                        bit_index = self.bit_index(i, j, wall)
                        self.board |= (1 << bit_index)

    def to_array(self):
        n = self.n
        npboard = np.zeros((n, n, 4), dtype=bool)
        for i in range(n):
            for j in range(n):
                for wall in range(4):
                    bit_index = self.bit_index(i, j, wall)
                    if (self.board & (1 << bit_index)):
                        npboard[i, j, wall] = 1
        return npboard

    def __getitem__(self, key):
        i, j, wall = key
        i = int(i)
        j = int(j)
        bit_index = self.bit_index(i, j, wall)
        return (int(self.board) & int(1 << bit_index)) != 0

    def walls(self, pos):
        i, j = pos
        # use bit shiftin to get the walls
        # we will need a bit mask , and then bit shift back
        i = int(i)
        j = int(j)
        bit_index = 4 * (self.n * i + j)
        # mask = 0xF << bit_index
        # walls = (self.board & mask) >> bit_index
        walls = (self.board >> bit_index) & 0xF

        return (bool(walls & 1), bool(walls & 2), bool(walls & 4), bool(walls & 8))

    def __str__(self) -> str:
        string = ""
        #print the bit board
        bit_string = bin(self.board)[2:]
        bit_string = "0" * (4 * self.n * self.n - len(bit_string)) + bit_string
        # print each cell by going through the row and column
        # but a space every 4 bits
        # using string slicing

        for i in range(self.n-1, -1, -1):

            for j in range(self.n-1, -1, -1):
                walls = bit_string[self.bit_index(i, j, 0):self.bit_index(i, j, 4)]
                string += walls[::-1] + " "
            string += "\n"
        string += "\n"
        return string


    def __setitem__(self, key, value):
        i, j, wall = key
        i = int(i)
        j = int(j)
        bit_index = self.bit_index(i, j, wall)

        if value:
            self.board |= (1 << bit_index)
        else:
            self.board &= ~(1 << bit_index)

    def __hash__(self):
        return hash(self.board)


class BitBoardTable():
    """
    This class is used to store win/loss data for a given board
    We could use a dictionary but their poorly optimized for this use case
    and we know the length of the key in advance
    """

    def __init__(self, n):
        self.n = n
        self.table = {} # (win, visits)

    def put(self, boardint, value):
        self.table[boardint][0] += value
        self.table[boardint][1] += 1

    def get(self, boardint):
        return self.table[boardint][0] / self.table[boardint][1]



def generate_random_board(n):
    import numpy as np
    board = np.zeros((n, n, 4), dtype=int)
    board[0, :, 0] = 1
    board[-1, :, 2] = 1
    board[:, 0, 3] = 1
    board[:, -1, 1] = 1

    for y in range(1, n - 1):
        for x in range(1, n - 1):
            for wall in range(4):
                if np.random.randint(0, 2):
                    board[y, x, wall] = 1

    return board

@lru_cache(maxsize=100000)
def h(current, adversary):

    dist =  (current[0] - adversary[0])**2 + (current[1] - adversary[1])**2
    # rount to 2 decimal places
    return dist

@lru_cache(maxsize=400)
def get_adjacent_moves(position, walls,
                       obstacle=None,
                       ):
    """
    This returns the list of directly adjacent positions to a given position
    Taking into account the walls and the adversary
    """

    moves = []
    deltas = [((-1, 0), 0), ((0, 1), 1), ((1, 0), 2), ((0, -1), 3)]
    x = position[0]
    y = position[1]
    for delta, i in deltas:
        nx, ny = x + delta[0], y + delta[1]
        if not walls[i] and (nx, ny) != obstacle:
            moves.append((nx, ny))
    return moves





def is_terminal(state, jump_point_search=False):
    """
    board is an mxmx4 grid, where m is the board size, and there are 4 wall positions
    note that to cells share a wall position if they are adjacent
    player is a tuple of (x, y) coordinates
    adversary is a tuple of (x, y) coordinates
    We are in a terminal state if there is no path from the player to the adversary
    This means that we have divided the board into two parts, and the adversary is in the part
    that the player cannot reach.
    we can do an A* search from the player to the adversary, and if we cannot find a path, then
    we are in a terminal state
    We actually dont need, to do a* , because we dont care about the path
    we just need to know if there is a path
    so we can use best first search
    """
    player = state['player']
    adversary = state['adversary']


    open_list = []
    heapq.heappush(open_list, (0, player))  # Heap element is a tuple (f_score, node)
    parent_node = {player: None}
    explored = set()

    while open_list:
        current = heapq.heappop(open_list)[1] # Get the node from the tuple

        if current in explored:
            continue

        explored.add(current)
        walls = state['board'].walls(current)
        neighbors = get_adjacent_moves(current, walls)

        for neighbor in neighbors:
            if neighbor == adversary:
                return False, explored
            if neighbor in explored:
                continue

            parent_node[neighbor] = current
            heapq.heappush(open_list, ( h(neighbor, adversary), neighbor))

    # No path found, it's a terminal state

    return True, explored

def count_closest_cells(adversary_distances, player_distances):
    """
    Count the number of cells that are closer to the player than the adversary.
    This takes a dict of distances from the adversary and a dict of distances from the player.
    It's the result of running Dijkstra's algorithm to calculate the shortest 
    path from the player and the adversary to all other cells.
    Returns:
        int: The difference in the number of cells that are closer to the player than the adversary.
    """
    # Count the number of cells that are closer to the player than the adversary
    closest_cells = 0
    for cell in player_distances:
        if cell in adversary_distances:
            if player_distances[cell] < adversary_distances[cell]:
                closest_cells += 1
            elif adversary_distances[cell] < player_distances[cell]:
                closest_cells -= 1

    return closest_cells

def simple_territory_search(board, player, adversary, max_step):
    """
    Do a bfs from both the player and the adversary and get their distance to every square
    in the players reachable positions 
    compare them against the adversary's distance to the same square
    if the player is closer, then the player controls that square
    if the adversary is closer, then the adversary controls that square
    """
    player_dists = get_possible_positions(board,
                                          max_step,
                                          player,
                                           depth_limited=False)
    adversary_dists = get_possible_positions(board,
                                              max_step,
                                              adversary,
                                              depth_limited=False)



    player_territory = {}
    adversary_territory = {}
    # go through the union of the keys of both dicts
    positions = set(player_dists.keys()).union(set(adversary_dists.keys()))
    overlap = set(player_dists.keys()).intersection(set(adversary_dists.keys()))

    for position in positions:
        # go through each position
        # if only the player can reach it, then the player controls it
        # if only the adversary can reach it, then the adversary controls it
        # if both can reach it, then we need to compare the distances
        # if the distance is equal, then they both control it
        if position not in player_dists:
            adversary_territory[position] = adversary_dists[position]
            continue
        if position not in adversary_dists:
            player_territory[position] = player_dists[position]
            continue
        if player_dists[position] < adversary_dists[position]:
            player_territory[position] = player_dists[position]
        elif adversary_dists[position] < player_dists[position]:
            adversary_territory[position] = adversary_dists[position]

    return player_territory, adversary_territory, overlap

def utility(state):
    """
    The utility of the board is used to mark how good the board is for the player
    There are a couple metrics we can use to determine the utility of the board
    - the number of cells the player can reach in their section
    - the number of cells the adversary can reach in their section
    - we might want to try to increase our number of reachable cells, while decreasing the adversary's number of reachable cells
    - we could use the distance between the player and the adversary
          - if we want to be aggressice, we would want to close the distance
          - if we want to be defensive, we would want to increase the distance
    - we could use the count of closest cells
      - that is we go through each cell in the space that is reachable from both the player and the adversary
      - we would count the distance to reach that cell from the player and the adversary
      - we would do this by performing dijkstra's algorithm from the player and then again from the adversary
      - then we would mark all cells that are closer to the player than the adversary as the player's cells
      - and all cells that are closer to the adversary than the player as the adversary's cells
      - we would then count the number of cells in each section
      - this is kind of like a territory control metric
    - we could have a metric for the number of walls that are continuous, this would make it more likely for the player to close walls of
    - could use q learning to learn a utility function
    """
    # New idea for implementation:
    # - use a dual BFS to determine territory control
    # - the utility is the difference in territory controlled by the player and the adversary
    p_t, a_t, overlap = simple_territory_search(state['board'], state['player'], state['adversary'], state['max_step'])
    if len(overlap) == 0:
        player_score = len(p_t)
        adversary_score = len(a_t)
        if player_score > adversary_score:
            return 1
        elif player_score < adversary_score:
            return -1
        else:
            return -0.99
    point_p = len(p_t)
    point_a = len(a_t)
    if point_p == 0 and point_a == 0:
        return 0
    win_priority_scaler = 0.5
    return ((point_p - point_a) / (point_p + point_a))  * win_priority_scaler    

def get_possible_positions(board,
                           max_step,
                           position,
                           depth_limited=True,
                           obstacle=None
                           ):
    """
    board is an mxmx4 grid, where m is the board size, and there are 4 wall positions
    note that to cells share a wall position if they are adjacent
    player is a tuple of (x, y) coordinates
    adversary is a tuple of (x, y) coordinates
    max_step is an integer that is the maxiumum manhattan distance a player can move
    is_player_turn is a boolean that is True if it is the player's turn, False otherwise
    To get the possible moves we want to perform a breadth first search from the player's position 
    to all reachable positions in max_step moves. We will use a queue to store the positions we
    need to visit, and a set to store the positions we have already visited.
    we cannot pass through walls and we cannot go through the adversary or on top of the adversary
    We also need to check if the board is in a game over state, if it is, we want to return an empty list
    """

    # we want to perform a breadth first search from the the player whos turn it is, to get to all reachable positions
    # that have a manhattan distance of max_step or less
    # The children of each node are the 4 adjacent positions
    # except those that are blocked by a wall, or the adversary

    # the queue should contain the position as well as the distance we travel to find that position
    # if the distance is greater than max_step, we do not want to add it to the queue

    init_pos = position
    queue = deque()
    distances = {}

    distances[init_pos] = 0
    queue.append(init_pos)

    while (len(queue) > 0):
        u = queue.popleft()
        if distances[u] >= max_step - 1 and depth_limited:
            continue
        walls = board.walls(u)
        for v in get_adjacent_moves(u, walls,
                                    obstacle=obstacle,
                                    ):
            if v not in distances:
                distances[v] = distances[u] + 1
                queue.append(v)

    return distances

def get_possible_moves(state):
    """
    This uses all the possible positions to get the possible moves
    A possible move is a position you can go where you can place a wall
    each position has 4 possible moves, one for each wall
    each placeable wall counts as a different move
    A move is a tuple of (position, wall)
    """
    terminal, explore = is_terminal(state)
    if terminal:
        return []
    pos = state['adversary']
    blocker = state['player'] 
    if state['is_player_turn']:
        pos = state['player'] 
        blocker = state['adversary'] 
    possible_positions = get_possible_positions(state['board'],state['max_step'], pos, obstacle=blocker)

    possible_moves = []
    for position in possible_positions:
        x = position[0]
        y = position[1]
        for i in range(4):
            if state['board'][x,y,i]:
                continue
            possible_moves.append((position, i))
    return possible_moves

def generate_children(state):
    """
    This generates the children of a given state
    An action that mutates the state of the board
    """
    possible_moves = get_possible_moves(state)
    return possible_moves

def perform_action(state, action):
    """
    mutates the state of the board given an action
    action = (new_position, wall)
    """
    position, wall = action
    x, y = position
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    # place wall, and opposite wall
    state['board'][x,y,wall] = True
    move = moves[wall]
    anti_x, anti_y = (x + move[0], y + move[1])
    state['board'][anti_x,anti_y,opposites[wall]] = True

    # for each action, we save whos turn it was , and where they move from
    # (turn, prev_position, action)

    # update player or adversary position

    if state['is_player_turn']:
        action_event = (state['is_player_turn'], state['player'], action)
        state['player'] = position
    else:
        action_event = (state['is_player_turn'], state['adversary'], action)
        state['adversary'] = position

    state['is_player_turn'] = not state['is_player_turn']
    state.get('action_history', []).append(action_event)

def undo_last_action(state):
    """
    undoes the action on the state
    """
    history = state.get('action_history', [])
    if len(history) == 0:
        return
    turn, prev_position, action = history.pop()
    if turn:
        state['player'] = prev_position
    else:
        state['adversary'] = prev_position

    position, wall = action
    x, y = position
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    # place wall, and opposite wall
    state['board'][x,y,wall] = False
    move = moves[wall]
    anti_x, anti_y = (x + move[0], y + move[1])
    state['board'][anti_x,anti_y,opposites[wall]] = False

    state['is_player_turn'] = not state['is_player_turn']

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

    class AlphaBeta:

        def get_action(generate_children, 
                    utility, 
                    state, 
                    max_depth, 
                    time_limit=0.5,
                    breadth_limit=10,
                    ):
            """
            children returned from generate children is a list of actiosn
            actions are (position, wall_direction)
            """
            start_time = time.time()

            def child_heuristic(child):
                player_pos = state['player']
                child_pos = child[0]
                adv_pos = state['adversary']
                distance_from_player = (player_pos[0] - child_pos[0])**2 + (player_pos[1] - child_pos[1])**2
                distance_to_adv = (adv_pos[0] - child_pos[0])**2 + (adv_pos[1] - child_pos[1])**2
                return distance_from_player * distance_to_adv  + distance_from_player

            def minimax(cur_state, depth, alpha, beta):
                # if max_player = True, then we are maximizing player
                current_time = time.time()
                if current_time - start_time > time_limit:
                    return utility(cur_state)
                children = generate_children(cur_state)
                children.sort(key=child_heuristic)
                children = children[:int(breadth_limit/depth)]


                if depth == max_depth or not children:
                    return utility(cur_state)

                max_player = cur_state['is_player_turn']

                cur_val = float('-inf') if max_player else float('inf')

                for child in children:
                    if current_time - start_time > time_limit:
                        break

                    perform_action(cur_state, child)
                    val = minimax(cur_state, depth + 1, alpha, beta)
                    undo_last_action(cur_state)

                    if max_player:
                        cur_val = max(cur_val, val)
                        alpha = max(alpha, cur_val)
                    else: 
                        cur_val = min(cur_val, val)
                        beta = min(beta, cur_val)
                    if beta <= alpha:
                        break
                return cur_val


            # get the move that maximizes the utility
            children = generate_children(state)
            # sort each child by its distance from the player

            # the sort function sorts in 
            children.sort(key=child_heuristic)
            children = children[:breadth_limit]

            max_child = children[0]
            max_val = float('-inf')
            for child in children:
                perform_action(state, child)
                child_val = minimax(state, 1, float('-inf'), float('inf'))
                undo_last_action(state)
                if child_val > max_val:
                    max_val = child_val
                    max_child = child

            def get_utility(move):
                perform_action(state, move)
                val = utility(state)
                undo_last_action(state)
                return val

            #print(f'chosen action: {max_child}, val: {max_val}, utility: {get_utility(max_child)}')
            if max_val == -1:
                for child in children:
                    perform_action(state, child)
                    child_val = utility(state)
                    undo_last_action(state)
                    if child_val > max_val:
                        max_val = child_val
                        max_child = child
                print(f'chosen action: {max_child}, val: {max_val} utility: {get_utility(max_child)}')

            return max_child

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

        if self.strategy == "AlphaBeta":
            new_action = self.AlphaBeta.get_action(
                generate_children,
                utility,
                state,
                self.kwargs.get('max_depth',2),
                self.kwargs.get('time_limit',1.0),
                self.kwargs.get('breadth_limit',400),
            )
        elif self.strategy == "Random":
            new_action = generate_children(state)[np.random.randint(len(generate_children(state)))]
            
        else:
            raise ValueError("Invalid strategy")


        time_taken = time.time() - start_time

        #print("My ALPHA-BETA AI's turn took ", time_taken, "seconds.")


        return new_action