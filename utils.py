import tensorflow as tf
import numpy as np

from game import pretty_print_board

def transform_state(x, player=1):
    n = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            for k in range(2):
                if x[i,j,k] != 0:
                    n[i, j] = (2*k - 1) * player
    return n


def ppp(i, phis, states):
    pretty_print_board(transform_state(states[i]), phis[i].flatten())


def ppmp(i, model, states):
    pretty_print_board(transform_state(states[i]), model.predict(states[[i]])[0].flatten())


def load(i): 
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    
    model = tf.keras.models.load_model(f"models/mini-zero-{i}")
    prev_model = tf.keras.models.load_model(f"models/mini-zero-{i-1}")
    phis = np.loadtxt(f"data/{i-1}/phis.npy", delimiter=",").reshape(-1, 100)
    vs = np.loadtxt(f"data/{i-1}/vals.npy", delimiter=",").reshape(-1, 1)
    states = np.loadtxt(f"data/{i-1}/states.npy", delimiter=",").reshape(-1, 10, 10, 2)

    return model, phis, vs, states
