import argparse
import tensorflow as tf
import numpy as np
import subprocess

from tqdm import trange
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import LearningRateScheduler

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
    pss = np.zeros(shape=phis.shape)
    sts = np.zeros(shape=states.shape)
    ks = np.random.choice(4, len(states), replace=True)
    for i, tup in enumerate(zip(ks, states, phis)):
        k, st, ps = tup
        pss[i] = np.rot90(ps.reshape(10, 10), k, axes=(0, 1)).reshape(100)
        sts[i] = np.rot90(st, k, axes=(0, 1))

    return sts, pss, vs


def remove_duplicates(states, phis, vs, p=0.9):
    def state_hash(s):
        return hash(s.tostring())

    seen_hashes = set()

    def seen(s):
        sh = state_hash(s)
        if sh in seen_hashes:
            return True
        seen_hashes.add(sh)
        return False
    indx = [
        i for i, s in enumerate(states) if not seen(s) or np.random.random() > p
    ]

    return states[indx], phis[indx], vs[indx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id')
    args = parser.parse_args()

    model_id = int(args.model_id)

    if model_id == 0:
        model = get_model((10, 10, 2), 100, layers_deep=3, kernel_size=5)
        model.save(f"models/mini-zero-{model_id}")
    else:
        model = tf.keras.models.load_model(f"models/mini-zero-{model_id - 1}")

        phis = np.loadtxt(f"data/{model_id - 1}/phis.npy", delimiter=",").reshape(-1, 100)
        vs = np.loadtxt(f"data/{model_id - 1}/vals.npy", delimiter=",").reshape(-1, 1)
        states = np.loadtxt(f"data/{model_id - 1}/states.npy", delimiter=",").reshape(-1, 10, 10, 2)
        
        epochs = 8
        mids = range(max(0, model_id - 10), model_id - 1)
        for mid in mids:
            _phis = np.loadtxt(f"data/{mid}/phis.npy", delimiter=",").reshape(-1, 100)
            _vs = np.loadtxt(f"data/{mid}/vals.npy", delimiter=",").reshape(-1, 1)
            _states = np.loadtxt(f"data/{mid}/states.npy", delimiter=",").reshape(-1, 10, 10, 2)

            _states, _phis, _vs = remove_duplicates(_states, _phis, _vs)
            phis = np.concatenate((phis, _phis), axis=0)
            vs = np.concatenate((vs, _vs), axis=0)
            states = np.concatenate((states, _states), axis=0)


        states, phis, vs = shuffle(states, phis, vs)
        states, phis, vs = add_rotations(states, phis, vs)

        idx = np.random.choice(phis.shape[0], phis.shape[0] // 4, replace=False)
        phis = phis[idx]
        vs = vs[idx]
        states = states[idx]

        def lr_scheduler(epoch, lr):
            if model_id < 50:
                return 1e-3
            return 1e-4

        model.fit(states, [phis, vs], epochs=epochs, batch_size=128, validation_split=0.1, callbacks=[LearningRateScheduler(lr_scheduler, verbose=0)])
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
