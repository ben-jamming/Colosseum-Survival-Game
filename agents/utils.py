import heapq
import numpy as np
from numpy import ndarray
from collections import deque

def get_possible_moves(board, player, adversary):
    pass

def h(current, adversary):
    # Manhattan distance
    return abs(current[0] - adversary[0]) + abs(current[1] - adversary[1])

def is_terminal(board: ndarray, player: tuple, adversary: tuple, get_possible_moves: function) -> bool:
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
    """
    open_list = []
    heapq.heappush(open_list, (0, player))  # Heap element is a tuple (f_score, node)
    parent_node = {player: None}
    g_score = {player: 0} # g: cheapest distance from start to node

    while open_list:
        current = heapq.heappop(open_list)[1] # Use a min heap
        if current == adversary:
            return False # We were able to reach the adversary so state not terminal

        for neighbor in get_possible_moves(board, current):
            # Generate the list of possible moves from the current node
            tentative_cheapest_distance = g_score[current] + 1 
            
            # Check if we have found a shorter path to the neighbor or if we haven't visited the neighbor yet
            if neighbor not in g_score or tentative_cheapest_distance < g_score[neighbor]:
                parent_node[neighbor] = current
                g_score[neighbor] = tentative_cheapest_distance
                f_score = tentative_cheapest_distance + h(neighbor, adversary)
                heapq.heappush(open_list, (f_score, neighbor))

    # No path found, it's a terminal state
    return True
    
def dijkstra(board, start, get_possible_moves):
    """
    Perform Dijkstra's algorithm to find the shortest path from start to all other reachable nodes.
    Returns a dict with nodes as keys and the shortest distance from start as values.
    """
    distances = {start: 0}
    open_list = [(0, start)]  # Priority queue

    while open_list:
        current_distance, current_node = heapq.heappop(open_list)

        for neighbor in get_possible_moves(board, current_node):
            distance = current_distance + 1
            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(open_list, (distance, neighbor))

    return distances

def dual_bfs_for_territory_search(board, player, adversary, get_possible_moves):

    """
    Perform a simultaneous BFS from both the player and the adversary to determine territory control.
    
    Args:
        board (ndarray): The game board.
        player (tuple): The current position of the player.
        adversary (tuple): The current position of the adversary.
        get_possible_moves (function): Function to get possible moves from a position.
        
    Returns:
        int: The net territory controlled by the player minus the territory controlled by the adversary.
    """
    # Initialize queues for BFS, starting from the player and adversary positions
    player_queue = deque([(player, 0)])  # Queue of tuples (position, distance from start)
    adversary_queue = deque([(adversary, 0)])
    
    # Set to keep track of visited positions to avoid re-visiting
    visited = set()
    
    # Variables to track the amount of territory controlled by player and adversary
    player_territory, adversary_territory = 0, 0

    # Continue BFS as long as there are positions to process in either queue
    while player_queue or adversary_queue:
        # Process the player's queue
        if player_queue:
            current_player, player_dist = player_queue.popleft()
            if current_player not in visited:  # Ensure each cell is processed only once
                visited.add(current_player)
                # Add all reachable positions from the current position to the queue
                for neighbor in get_possible_moves(board, current_player):
                    if neighbor not in visited:
                        player_queue.append((neighbor, player_dist + 1))

        # Process the adversary's queue in a similar manner
        if adversary_queue:
            current_adversary, adversary_dist = adversary_queue.popleft()
            if current_adversary not in visited:
                visited.add(current_adversary)
                for neighbor in get_possible_moves(board, current_adversary):
                    if neighbor not in visited:
                        adversary_queue.append((neighbor, adversary_dist + 1))

        # Check for territory control
        # The player or adversary with the shorter distance to a cell claims its territory
        if player_dist < adversary_dist:
            player_territory += 1
        elif adversary_dist < player_dist:
            adversary_territory += 1

    # Utility is the difference in territory controlled by the player and the adversary
    return player_territory - adversary_territory

def utility(board, player, adversary, max_step, is_player_turn):
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
    return dual_bfs_for_territory_search(board, player, adversary, get_possible_moves)



