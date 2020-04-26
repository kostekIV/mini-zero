import argparse
import tensorflow as tf
import numpy as np
import subprocess

from tqdm import trange
from sklearn.utils import shuffle

from model import get_model


def bootstrap_from_to(fr, to):
    model = get_model((10, 10, 2), 100, 1, kernel_size=5)
    for i in range(fr, to):
        phis = np.loadtxt(f"data/{i}/phis.npy", delimiter=",").reshape(-1, 100)
        vs = np.loadtxt(f"data/{i}/vals.npy", delimiter=",").reshape(-1, 1)
        states = np.loadtxt(f"data/{i}/states.npy", delimiter=",").reshape(-1, 10, 10, 2)

        phis, vs, states = shuffle(phis, vs, states)
        model.fit(states, [phis, vs], epochs=20, batch_size=64, validation_split=0.1)
        model.save(f"models/mini-zero-{to}")


def add_rotations(states, phis, vs):
    vs = np.tile(vs, (2, 1))
    pss = []
    sts = []
    ks = np.random.choice(4, 2, replace=False)
    for k in ks:
        pss += [np.rot90(phis.reshape(-1, 10, 10), k, axes=(1, 2)).reshape(-1, 100)]
        sts += [np.rot90(states, k, axes=(1, 2))]
    states = np.concatenate(sts, axis=0)
    phis = np.concatenate(pss, axis=0)

    return states, phis, vs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id')
    args = parser.parse_args()

    model_id = int(args.model_id)

    if model_id == 0:
        model = get_model((10, 10, 3), 100, 1, kernel_size=5)
        model.save(f"models/mini-zero-{model_id}")
    else:
        model = tf.keras.models.load_model(f"models/mini-zero-{model_id - 1}")

        phis = np.loadtxt(f"data/{model_id - 1}/phis.npy", delimiter=",").reshape(-1, 100)
        vs = np.loadtxt(f"data/{model_id - 1}/vals.npy", delimiter=",").reshape(-1, 1)
        states = np.loadtxt(f"data/{model_id - 1}/states.npy", delimiter=",").reshape(-1, 10, 10, 3)
        
        epochs = 12
        # Get rid of begginers model relatively fast
        # but later keep more datas from old games to disallow overfitting to current network
        if model_id >= 2:
            if model_id < 5:
                mids = range(0, model_id - 1)
            elif model_id < 15:
                mids = range(max(1, model_id - 7), model_id - 1)
            else:
                mids = range(max(9, model_id - 10), model_id - 1)
            for mid in mids:
                phis = np.concatenate((phis, np.loadtxt(f"data/{mid}/phis.npy", delimiter=",").reshape(-1, 100)), axis=0)
                vs = np.concatenate((vs, np.loadtxt(f"data/{mid}/vals.npy", delimiter=",").reshape(-1, 1)), axis=0)
                states = np.concatenate((states, np.loadtxt(f"data/{mid}/states.npy", delimiter=",").reshape(-1, 10, 10, 3)), axis=0)


        states, phis, vs = add_rotations(states, phis, vs)
        states, phis, vs = shuffle(states, phis, vs)

        idx = np.random.choice(phis.shape[0], phis.shape[0] // 4, replace=False)
        phis = phis[idx]
        vs = vs[idx]
        states = states[idx]

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
    main()
