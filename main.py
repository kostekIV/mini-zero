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

    if model_id == 0:
        model = get_model((10, 10, 2), 100)
        model.save(f"models/mini-zero-{model_id}")
    else:
        model = tf.keras.models.load_model(f"models/mini-zero-{model_id - 1}")

        phis = np.loadtxt(f"data/{model_id - 1}/phis.npy", delimiter=",").reshape(-1, 100)
        vs = np.loadtxt(f"data/{model_id - 1}/vals.npy", delimiter=",").reshape(-1, 1)
        states = np.loadtxt(f"data/{model_id - 1}/states.npy", delimiter=",").reshape(-1, 10, 10, 2)
        
        ids = [i for i in range(model_id - 1)][-50:]
        if len(ids) > 0:
            second_ids = np.random.choice(ids, min(len(ids), 5), replace=False)
            for m_id in second_ids:
                phis = np.concatenate((phis, np.loadtxt(f"data/{m_id}/phis.npy", delimiter=",").reshape(-1, 100)), axis=0)
                vs = np.concatenate((vs, np.loadtxt(f"data/{m_id}/vals.npy", delimiter=",").reshape(-1, 1)), axis=0)
                states = np.concatenate((states, np.loadtxt(f"data/{m_id}/states.npy", delimiter=",").reshape(-1, 10, 10, 2)), axis=0)

        epochs = 10

        phis, vs, states = shuffle(phis, vs, states)
        model.fit(states, [phis, vs], epochs=epochs, batch_size=64, validation_split=0.1)
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
    print(gpus)
    main()
