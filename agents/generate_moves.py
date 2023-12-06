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

import itertools
boolean_values = [True, False]
combinations = list(itertools.product(boolean_values, repeat=4))

print(f'adjacents = {{')
for i in range(0,12):
    for j in range(0,12):
        for wall_combo in combinations:
            moves = get_adjacent_moves((i,j), wall_combo)
            print(f'({(i,j)},{wall_combo}): {moves},')
print(f'}}')