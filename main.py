import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id')
    args = parser.parse_args()

    model_id = int(args.model_id)
    model = get_model((9, 9, 2), 81)

    if model_id == 0:
        model.save(f"models/mini-zero-{model_id}")
    else:
        phis = np.load(f"data/{model_id - 1}/phis.npy")
        vs = np.load(f"data/{model_id - 1}/vals.npy")
        states = np.load(f"data/{model_id - 1}/states.npy")

        if model_id > 10:
            a = np.arange(model_id - 1)[-10:]
            ids = np.random.choice(a, 5, replace=False)
            for id in ids:
                phis = np.concatenate((phis, np.load(f"data/{id}/phis.npy")), axis=0)
                vs = np.concatenate((vs, np.load(f"data/{id}/vals.npy")), axis=0) 
                states = np.concatenate((states, np.load(f"data/{id}/states.npy")), axis=0)

        model.fit(states, [phis, vs])
        model.save(f"models/mini-zero-{model_id}")


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
