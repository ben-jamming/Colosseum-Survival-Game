import heapq
import numpy as np
from numpy import ndarray
from collections import deque
import random
from functools import lru_cache
from .jp_search import find_jump_points

@lru_cache(maxsize=100000)
def h(current, adversary):

    dist =  (current[0] - adversary[0])**2 + (current[1] - adversary[1])**2
    # rount to 2 decimal places
    return dist

def get_adjacent_moves(position, state,
                       obstacles=[],
                       ):
    """
    This returns the list of directly adjacent positions to a given position
    Taking into account the walls and the adversary
    """

    moves = []
    deltas = [((-1, 0), 0), ((0, 1), 1), ((1, 0), 2), ((0, -1), 3)]
    x = position[0]
    y = position[1]
    walls = state['board'][x][y]
    for delta, i in deltas:
        nx, ny = x + delta[0], y + delta[1]
        if not walls[i] and (nx, ny) not in obstacles:
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
        neighbors = get_adjacent_moves(current, state)

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

def simple_territory_search(state):
    """
    Do a bfs from both the player and the adversary and get their distance to every square
    in the players reachable positions 
    compare them against the adversary's distance to the same square
    if the player is closer, then the player controls that square
    if the adversary is closer, then the adversary controls that square

    """
    player_dists = get_possible_positions(state,
                                          state['player'],
                                           depth_limited=False)
    adversary_dists = get_possible_positions(state,
                                              state['adversary'],
                                              depth_limited=False)



    player_territory = {}
    adversary_territory = {}
    # go through the union of the keys of both dicts
    positions = set(player_dists.keys()).union(set(adversary_dists.keys()))

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

    return player_territory, adversary_territory

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
    p_t, a_t = simple_territory_search(state)
    # if p_t and a_t do not share any keys, then we are in a terminal state
    if len(set(p_t.keys()).intersection(set(a_t.keys()))) == 0:
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


def score(state)-> (float, float) :
    """
    Once we know the game is over, we need to determine the score of each player
    The score is given by the number of cells that the player can reach in their section
    The adversary's score is given by the number of cells that the adversary can reach in their section

    To calculate this we can perform a breadth first search from the player's position to all reachable positions
    in the player's section. We will use a queue to store the positions we need to visit, and a set to store the
    positions we have already visited.

    We then can do the same for the adversary's section.
    """

    player_score = len(get_possible_positions(state,
                                              state['player'],
                                              depth_limited=False))
                                              
    adversary_score = len(get_possible_positions(state, 
                                                  state['adversary'],
                                                 depth_limited=False))
    return player_score, adversary_score
    

def get_possible_positions(state,
                           position,
                           depth_limited=True,
                           obstacles=[]
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
    queue = []
    distances = {}

    distances[init_pos] = 0
    queue.append(init_pos)

    while (len(queue) > 0):
        u = queue.pop(0)
        if distances[u] >= state['max_step'] - 1 and depth_limited:
            continue
        for v in get_adjacent_moves(u, state, 
                                    obstacles=obstacles,
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
    # terminal, explore = is_terminal(state)
    # if terminal:
    #     return []
    if state['is_player_turn']:
        possible_positions = get_possible_positions(state,
                                                    state['player'],
                                                    obstacles=[state['adversary']]
                                                    )
    else:
        possible_positions = get_possible_positions(state,
                                                    state['adversary'],
                                                    obstacles=[state['player']]
                                                    )
    
    possible_moves = []
    for position in possible_positions:
        x = position[0]
        y = position[1]
        for i in range(4):
            if state['board'][x][y][i]:
                continue
            possible_moves.append((position, i))
    return possible_moves

def generate_children(state):
    """
    This generates the children of a given state
    An action that mutates the state of the board
    """
    possible_moves = get_possible_moves(state)
    if len(possible_moves) == 0:
        return []
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
    state['board'][x][y][wall] = True
    move = moves[wall]
    anti_x, anti_y = (x + move[0], y + move[1])
    state['board'][anti_x][anti_y][opposites[wall]] = True

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
    state['board'][x][y][wall] = False
    move = moves[wall]
    anti_x, anti_y = (x + move[0], y + move[1])
    state['board'][anti_x][anti_y][opposites[wall]] = False

    state['is_player_turn'] = not state['is_player_turn']


def is_possible_move(state, start_pos, new_pos):
    """
    checks if a position is a possible move
    to do this we perform best first search, but stop if we go above the max_step
    if we go above the max step we return false and stop the search
    """
    # we want to perform a breadth first search from the the player whos turn it is, to get to all reachable positions
    # that have a manhattan distance of max_step or less
    # The children of each node are the 4 adjacent positions
    # except those that are blocked by a wall, or the adversary

    # the queue should contain the position as well as the distance we travel to find that position
    # if the distance is greater than max_step, we do not want to add it to the queue

    open_list = []
    heapq.heappush(open_list, (0, start_pos))  # Heap element is a tuple (f_score, node)
    parent_node = {start_pos: None}
    explored = set()
    cost = {start_pos: 0}

    while open_list:
        # print("2looping")
        current = heapq.heappop(open_list)[1] # Get the node from the tuple

        if cost[current] > state['max_step']:
            continue
        if current in explored:
            continue
        if current == new_pos:
            return True
        
        
        explored.add(current)


        neighbors = get_adjacent_moves(current, state)
        for neighbor in neighbors:
            if neighbor in explored:
                continue
            explored.add(neighbor)
            parent_node[neighbor] = current
            cost[neighbor] = cost[current] + 1
            if cost[neighbor] <= state['max_step']:
                heapq.heappush(open_list, ( h(neighbor, new_pos), neighbor)) 

    return False

        
    

# def mcts_get_random_move(state):
#     """
#     usually we generate the children and then pick one at random
#     but this is slow and wastes alot of time
#     instead we can pick a random move in a range around the player/ or adversary
#     then we do a best first search from that random position to the player
#     If its not, then we pick a different one
#     """
#     player_pos = state['player']
#     adversary_pos = state['adversary']
#     turn = state['is_player_turn']

#     pos = player_pos if turn else adversary_pos

#     # max step is the max distance we can move
#     # so our range is pos - max_step to pos + max_step
#     attempted_moves = set()
#     attempts = 0
#     random_move_found = False
#     x_bounds = (pos[0] - state['max_step'], pos[0] + state['max_step'])
#     y_bounds = (pos[1] - state['max_step'], pos[1] + state['max_step'])
#     # make sure its in bounds of board
#     board =state['board']
#     board_len_x = len(board)
#     board_len_y = len(board[0])
#     x_bounds = (max(0, x_bounds[0]), min(board_len_x - 1, x_bounds[1]))
#     y_bounds = (max(0, y_bounds[0]), min(board_len_y - 1, y_bounds[1]))

#     # hash all combination of x and y so that their unique and randomize

#     # then try them in order



#     while not random_move_found:
#         attempts += 1
#         # print("1looping")
#         x = random.randint( x_bounds[0], x_bounds[1])
#         y = random.randint( y_bounds[0], y_bounds[1])
#         random_move = (x, y)
#         # print(f'board n : {board_len_x * board_len_y}')
#         # print(f'attempts: {attempts}')
#         if attempts > board_len_x * board_len_y:
#             return None
#         if random_move in attempted_moves:
#             attempted_moves.add(random_move)
#             continue
#         manhattan_distance = abs(x - pos[0]) + abs(y - pos[1])
#         if manhattan_distance > state['max_step']:
#             attempted_moves.add(random_move)
#             continue
#         for i in range(4):
#             if not state['board'][x][y][i]: # and is_possible_move(state, pos, random_move):
#                 random_move_found = True
#                 move = (random_move, i)
#                 break
        


#     return move
# import hashlib
# def mcts_get_random_move(state):
#     player_pos = state['player']
#     adversary_pos = state['adversary']
#     turn = state['is_player_turn']

#     pos = player_pos if turn else adversary_pos
#     max_step = state['max_step']
#     board_len_x = len(state['board'])
#     board_len_y = len(state['board'][0])

#     # Calculate bounds
#     x_bounds = (max(0, pos[0] - max_step), min(board_len_x - 1, pos[0] + max_step))
#     y_bounds = (max(0, pos[1] - max_step), min(board_len_y - 1, pos[1] + max_step))

#     # Total number of possible moves
#     total_moves = (x_bounds[1] - x_bounds[0] + 1) * (y_bounds[1] - y_bounds[0] + 1)
#     seed = "fixed_seed"  # Fixed seed for deterministic hashing

#     for i in range(total_moves):
#         # Generate a hash value
#         hash_value = int(hashlib.md5(f"{seed}_{i}".encode()).hexdigest(), 16)
#         x = hash_value % (x_bounds[1] - x_bounds[0] + 1) + x_bounds[0]
#         y = (hash_value // (x_bounds[1] - x_bounds[0] + 1)) % (y_bounds[1] - y_bounds[0] + 1) + y_bounds[0]

#         random_move = (x, y)
#         manhattan_distance = abs(x - pos[0]) + abs(y - pos[1])

#         if manhattan_distance <= max_step:
#             for wall_index in range(4):
#                 if not state['board'][x][y][wall_index]:  # Check if move is possible
#                     return (random_move, wall_index)

#     return None  # No valid move found
    

    

        