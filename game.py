import numpy as np

from copy import deepcopy


def pretty_print_board(board, probs=None):
    mp = {
        1: '\033[92mX\033[0m',
        -1: '\033[94mO\033[0m',
        0: ' '
    }

    n, m = board.shape
    b = [
        [mp[board[i, j]] for j in range(m)] for i in range(n)
    ]

    if probs is not None:
        for i, row in enumerate(b):
            for j, x in enumerate(row):
                if x == ' ':
                    b[i][j] = f"{probs[i * m + j]:.3f}"
    cell_w = 7
    print("-"*(cell_w*m +m))
    for row in b:
        row_str = "|"
        row_str += (" " * cell_w + '|') * m
        print(row_str)
        x_row = "|"
        for x in row:
            if x in "XO ":
                x_row += f"   {x}   |"
            else:
                x_row += f" {x} |"
        print(x_row)
        print(row_str)
        print("-"*(cell_w*m +m))


class GameState:
    WIN = 0
    TIE = 1
    IN_PROGRESS = 2

class TicTacToe():
    def __init__(self, N, M, win_req):
        self.win_req = win_req
        self.size = (N, M)
        self.board = np.zeros(self.size)

        self.winner = None
        self.turn = 1
        self.history = []

    def _map_moves(self, moves):
        a = np.zeros(self.size[0] * self.size[1])
        for i, j in moves:
            a[i*self.size[1] + j] = 1
        return a

    def allowed_moves(self):
        moves =  [
            (i, j) for i, j in zip(*(self.board == 0).nonzero())
        ]
        return self._map_moves(moves)

    def state(self):
        return deepcopy(self.board)

    def move(self, k):
        i = k // self.size[1]
        j = k % self.size[1]
        if self.board[i, j] != 0:
            raise ValueError(f"Invalid move [{i}][{j}] already taken")

        self.history.append((self.turn, (i, j)))
        self.board[i, j] = self.turn

        self.turn *= -1
        return self._check_winner()

    def _check_winner(self):
        player, last_move = self.history[-1]

        i, j = last_move
        in_row = 0

        # check horizontal and vertical
        end = j
        while end < self.size[1] and self.board[i, end] == player:
            end += 1
        start = j
        while start >= 0 and self.board[i, start] == player:
            start -= 1
        in_row = max(end - start - 1, in_row)

        end = i
        while end < self.size[0] and self.board[end, j] == player:
            end += 1
        start = i
        while start >= 0 and self.board[start, j] == player:
            start -= 1
        in_row = max(end - start - 1, in_row)

        # diagonals
        end_i = i
        end_j = j
        while end_i < self.size[0] and end_j < self.size[1] and self.board[end_i, end_j] == player:
            end_i += 1
            end_j += 1

        start_i = i
        start_j = j
        while start_i >= 0 and start_j >= 0 and self.board[start_i, start_j] == player:
            start_i -= 1
            start_j -= 1
        in_row = max(end_i - start_i - 1, in_row)

        end_i = i
        end_j = j
        while end_i < self.size[0] and end_j >= 0 and self.board[end_i, end_j] == player:
            end_i += 1
            end_j -= 1

        start_i = i
        start_j = j
        while start_i >= 0 and start_j < self.size[1] and self.board[start_i, start_j] == player:
            start_i -= 1
            start_j += 1
        in_row = max(end_i - start_i - 1, in_row)

        if in_row >= self.win_req:
            self.winner = player
            return GameState.WIN

        if len(self.allowed_moves()) == 0:
            self.winner = 0
            return GameState.TIE

        return GameState.IN_PROGRESS

    def in_progress(self):
        return self.winner is None