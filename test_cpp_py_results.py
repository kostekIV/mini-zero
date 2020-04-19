import numpy as np

import argparse
import tensorflow as tf
import subprocess

from model import get_model

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

parser = argparse.ArgumentParser()
parser.add_argument('model_id')
args = parser.parse_args()

model_id = int(args.model_id)


states = np.random.rand(200).reshape(1, 10, 10, 2)


model = tf.keras.models.load_model(f"models/mini-zero-{model_id}")

result = model.predict(states)[0]


with open("test.in", "w") as t:
    print(200, file=t)
    for x in states.flatten():
        print(x, sep=',', file=t)


with open("test.in", "r") as _in, open("result_cpp", "w") as out:
    subprocess.run(["./pred", str(model_id)], stdout=out, stdin=_in)

with open("result_cpp", "r") as cpp:
    s = cpp.read().strip().split(',')[:-1]

    for c, p in zip(s, result.flatten()):
        if abs(float(c) - p) > 1e-5:
            print(abs(float(c) - p))
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
z = np.zeros((1, 10, 10, 2))
z[0, 0, 0, 1] = 1
p, v = model.predict(np.zeros((1, 10, 10, 2)))
p2, v2 = model.predict(z)
z[0, 1, 1, 0] = 1

p3, v3 = model.predict(z)
print(p.reshape(10, 10), v)
print(p2.reshape(10, 10), v2)
print(p3.reshape(10, 10), v3)