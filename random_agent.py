import random


class RandomAgent:
    def move(self, game):
        allowed_moves = game.allowed_moves()
        move = random.randint(0, len(allowed_moves)-1)
        while allowed_moves[move] == 0:
            move = random.randint(0, len(allowed_moves)-1)
        return move, None, None
