import tensorflow as tf
import numpy as np

from tqdm import trange
from sklearn.utils import shuffle

from random_agent import RandomAgent as RA
from mcts_agent import MCTSAgent as MA
from game import TicTacToe as TTT, pretty_print_board
from model import get_model


def play_one(game, a, b, s=0, verbose=False):
    players = [a, b]
    history = []
    turn = 0
    while game.in_progress():
        p = players[turn]
        k, phi, state = p.move(game)
        history.append([state, phi, None])
        if verbose:
            pretty_print_board(game.state(), phi)
        game.move(k)
        turn = (turn + 1) % 2
    winner = game.winner
    for i in range(len(history)):
        history[i][2] = winner
        winner *= -1
    if verbose:
        pretty_print_board(game.state(), phi)
        print(f"winner {game.winner}")
    return history[s::2]

def main():
    model = get_model((9, 9, 2), 81)
    h = []
    for i in range(1000):
        for j in trange(10, desc='models are playing :)'):
            t = TTT(9, 9, 5)
            players = [None, None]
            players[i%2] = MA(model, 101)
            players[(i+1)%2] = MA(model, 50)
            h += play_one(t, *players, j % 2, False)
        
        train, phis, vs = shuffle(
            np.array([x[0] for x in h]),
            np.array([x[1] for x in h]),
            np.array([x[2] for x in h])
        )
        n66 = len(train) // 3 * 2
        model.fit(train[:n66], [phis[:n66], vs[:n66]])
        h = shuffle(h)[-10000:] # keep some old records
        if i % 10 == 0:
            t = TTT(9, 9, 5)
            players = [MA(model), MA(model)]
            play_one(t, *players, i % 2, True)
    model.save("models/mini-zero")


if __name__ == '__main__':
    # cudnn problem
    tf.get_logger().setLevel('WARNING')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    main()
