import heapq
from collections import deque
import random
from functools import lru_cache

@lru_cache(maxsize=1000)
def h(current, adversary):

    dist =  (current[0] - adversary[0])**2 + (current[1] - adversary[1])**2
    # rount to 2 decimal places
    return dist

@lru_cache(maxsize=3000)
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

@lru_cache(maxsize=1000)
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

    # cheap utility, return a negative number of the walls directly adjacent to the player
    # this will encourage the player to move away from the walls
    # return -len(state['board'].walls(state['player'])) / 8  + len(state['board'].walls(state['adversary'])) / 8

    # return the euclidean distance between the player and the adversary

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
    point_p = len(p_t) * len(p_t)
    point_a = len(a_t) * len(a_t) * 2
    if point_p == 0 and point_a == 0:
        return 0
    win_priority_scaler = 0.5
    return ((point_p - point_a) / (point_p + point_a))  * win_priority_scaler    


@lru_cache(maxsize=1000)
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

def get_possible_moves(state, terminal_check=True):
    """
    This uses all the possible positions to get the possible moves
    A possible move is a position you can go where you can place a wall
    each position has 4 possible moves, one for each wall
    each placeable wall counts as a different move
    A move is a tuple of (position, wall)
    """
    if terminal_check:
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

def generate_children(state, terminal_check=True):
    """
    This generates the children of a given state
    An action that mutates the state of the board
    """
    possible_moves = get_possible_moves(state, terminal_check=terminal_check)
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



def mcts_get_random_move(state):
    """
    pick a random position within the range of max step, and a random wall 
    """
    player_pos = state['player']
    adversary_pos = state['adversary']
    turn = state['is_player_turn']

    pos = player_pos if turn else adversary_pos

    # max step is the max distance we can move
    # so our range is pos - max_step to pos + max_step

    random_move_found = False
    x_bounds = (pos[0] - state['max_step'], pos[0] + state['max_step'])
    y_bounds = (pos[1] - state['max_step'], pos[1] + state['max_step'])
    # if there is wall in the direction of the extended bound, cut it off to just the position
    for i in range(4):
        if state['board'][pos[0], pos[1], i]:
            if i == 0:
                x_bounds = (pos[0], x_bounds[1])
            elif i == 1:
                y_bounds = (y_bounds[0], pos[1])
            elif i == 2:
                x_bounds = (x_bounds[0], pos[0])
            elif i == 3:
                y_bounds = (pos[1], y_bounds[1])
    # make sure its in bounds of board
    board =state['board']
    x_bounds = (max(0, x_bounds[0]), min(board.n - 1, x_bounds[1]))
    y_bounds = (max(0, y_bounds[0]), min(board.n - 1, y_bounds[1]))


    random_move_found = True
    count = 0
    while not random_move_found:
        if count > board.n * board.n :
            return None
        move = random.randint(0, board.n - 1), random.randint(0, board.n - 1)
        for i in range(4):
            if not state['board'][move[0], move[1], i]:
                random_move_found = True
                return (move, i)
        count += 1