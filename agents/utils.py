

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
    pass

def score(board, player, adversary):
    """
    Once we know the game is over, we need to determine the score of each player
    The score is given by the number of cells that the player can reach in their section
    The adversary's score is given by the number of cells that the adversary can reach in their section

    To calculate this we can perform a breadth first search from the player's position to all reachable positions
    in the player's section. We will use a queue to store the positions we need to visit, and a set to store the
    positions we have already visited.

    We then can do the same for the adversary's section.
    """
    pass

def is_terminal(board, player, adversary):
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
    pass



def get_possible_moves(board, player, adversary, max_step, is_player_turn):
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
    pass

