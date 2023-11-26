import heapq

def get_possible_moves(board, player, adversary):
    pass

def h(current, adversary):
    # Manhattan distance
    return abs(current[0] - adversary[0]) + abs(current[1] - adversary[1])

def is_terminal(board, player, adversary, get_possible_moves: function) -> bool:
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
                f_score = tentative_cheapest_distance + h(neighbor)
                heapq.heappush(open_list, (f_score, neighbor))

    # No path found, it's a terminal state
    return True

    

