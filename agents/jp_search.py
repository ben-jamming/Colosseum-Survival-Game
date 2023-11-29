

# deltas = [((-1, 0), 0), ((0, 1), 1), ((1, 0), 2), ((0, -1), 3)]
deltas = {
    (-1, 0): 0, # up
    (0, 1): 1, # right
    (1, 0): 2, # down
    (0, -1): 3, # left
}


# IN THIS BOARD, X VERTICAL AND Y HORIZONTAL!!!!!!!!!!!!!!

def is_jump_point(state, pos, direction):
    """
    check for forced neighbours
    """
    x, y = pos
    dx, dy = direction

    # dx is vertical
    # dy is horizontal

    adv_point = state['adversary']
    # Get the wall index for the given direction
    wall_index = deltas[(dx, dy)]

    # Check if there is a wall in the direction of movement
    if state['board'][x][y][wall_index]:
        return True

    # Additional check for adversary's alignment
    if adv_point[0] == x and adv_point[1] == y:
        return True

    # Check for forced neighbours based on adjacent walls
    # For horizontal movement, check vertical walls and vice versa
    if dx != 0:  # Moving vertically
        # Check walls to the left and right of the current position
        if (y > 0 and state['board'][x][y-1][1]) or (y < len(state['board'][0]) - 1 and state['board'][x][y+1][1]):
            return True
    elif dy != 0:  # Moving horizontally
        # Check walls above and below the current position
        if (x > 0 and state['board'][x-1][y][2]) or (x < len(state['board']) - 1 and state['board'][x+1][y][2]):
            return True
    

    return False

def search_direction(state, pos, direction):
    """
    @param pos: Start position of the horizontal scan. 
    @param hor_dir: Horizontal direction (+1 or -1). 
    @param dist: Distance traveled so far. 
    @return: New jump point nodes (which need a parent). 
    """

    x, y = pos
    dx, dy = direction
    jump_points = []

    while True:
        # check if there is a wall in the direcion we are going
        # check if there is a wall inbetween x and x + dx, and y and y + dy
        if state['board'][x][y][deltas[(dx, dy)]]:
            break

        x += dx
        y += dy

        if x < 0 or x >= len(state['board']) or y < 0 or y >= len(state['board'][0]):
            break

        if is_jump_point(state, (x, y), direction):
            jump_points.append((x, y))
            break

        
    return jump_points

def find_jump_points(state, pos):
    """
    @param pos: Start position of the horizontal scan. 
    @param hor_dir: Horizontal direction (+1 or -1). 
    @param dist: Distance traveled so far. 
    @return: New jump point nodes (which need a parent). 
    """
    jump_points = []

    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                  # (-1, 1), (1, 1), (1, -1), (-1, -1)]

    # sort direction that move towards the adversary first
    # this will help us find the jump point faster
    adv_point = state['adversary']
    def distance_to_adv(direction):
        dx, dy = direction
        pos_x, pos_y = pos
        pos_x += dx
        pos_y += dy
        return abs(pos_x - adv_point[0]) + abs(pos_y - adv_point[1])
    directions.sort(key=distance_to_adv)
    # we want the onse thate are most similar to the direction of the adversary

    
    for direction in directions:
        new_jump_points = search_direction(state, pos, direction)
        jump_points.extend(new_jump_points)
        # if jump_points:
        #     break
    
    return jump_points
        



def jumpPointSearch(state):
    """
    We want to use jump point searchto check if it is a terminal state
    We do this by try to see if there is a path between the player and the adversary
    if there is a path, then it is not a terminal state
    if there is no path, then it is a terminal state
    Our jump point search will have to be slightly modified
    We are on a grid, and instead of obstacle cells
    each cell has 4 walls, one for each direction
    we cannot pass through walls
    Also we cannot move diagonaly, we can only move up, down, left, or right
    """