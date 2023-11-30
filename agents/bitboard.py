import numpy as np

import matplotlib.pyplot as plt

class BitBoard():

    def __init__(self, npboard=None):
        if npboard is not None:
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
                        self.board |= 1 << (4 * (self.n * i + j) + wall)
    
    def to_array(self):
        n = self.n
        npboard = np.zeros((n, n, 4), dtype=bool)
        for i in range(n):
            for j in range(n):
                for wall in range(4):
                    bit_index = (4 * (self.n * i + j) + wall)
                    if (self.board & (1 << bit_index)):
                        npboard[i, j, wall] = 1
        return npboard
    
    def __getitem__(self, key):
        i, j, wall = key
        bit_index = (4 * (self.n * i + j) + wall)
        return (self.board & (1 << bit_index)) != 0
    
    def __setitem__(self, key, value):
        i, j, wall = key
        bit_index = (4 * (self.n * i + j) + wall)
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


def display_board(board):
    """
    a bord is a grid of cells with 4 walls each
    we want to display each cell with its walls
    walls should be a distinct color
    """

    n = board.shape[0]
    fig, ax = plt.subplots(figsize=(n, n))
    wall_color = 'black'

    for y in range(n):
        for x in range(n):
            # rect = plt.Rectangle((x, y), 1, 1, facecolor='none', edgecolor='gray')
            # ax.add_patch(rect)

            if board[y, x, 0]:
                ax.plot([x, x + 1], [y, y], color=wall_color)
            if board[y, x, 1]:
                ax.plot([x + 1, x + 1], [y, y + 1], color=wall_color)
            if board[y, x, 2]:
                ax.plot([x + 1, x], [y + 1, y + 1], color=wall_color)
            if board[y, x, 3]:
                ax.plot([x, x], [y + 1, y], color=wall_color)

if __name__ == "__main__":
    npboard = generate_random_board(10)
    board = BitBoard(npboard)

    # cover the first top half of the board with all walls using bit
    for i in range(5):
        for j in range(10):
            board[i, j, 0] = 1
            board[i, j, 1] = 1
            board[i, j, 2] = 1
            board[i, j, 3] = 1
    
    # speed test bit borad read and write vs numpy array
    iterations = 1000000
    import time
    start = time.time()
    for i in range(iterations):
        board[0, 0, 0] = 1
        board[0, 0, 1] = 1
        board[0, 0, 2] = 1
        board[0, 0, 3] = 1
    end = time.time()

    print("bitboard time: ", end - start)

    start = time.time()
    for i in range(iterations):
        npboard[0, 0, 0] = 1
        npboard[0, 0, 1] = 1
        npboard[0, 0, 2] = 1
        npboard[0, 0, 3] = 1
    end = time.time()

    print("numpy array time: ", end - start)


    board = board.to_array()
    display_board(board)
    plt.show()