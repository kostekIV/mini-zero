import pathlib
import subprocess
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('model_id')
args = parser.parse_args()

model_id = int(args.model_id)


subprocess.run("g++ cpp/main.cpp cpp/model.cpp cpp/game.cpp -L/usr/local/lib -lgsl -ltensorflow -o main -std=c++17 -O3".split())
while True:

    # if model_id <= 4:
    #     second_ids = [i for i in range(model_id)]
    # else:
    #     # only last 50 models
    #     ids = [i for i in range(model_id)][-50:]
    #     second_ids = np.random.choice(ids, 5, replace=False)

    pathlib.Path(f'./data/{model_id}').mkdir(parents=True, exist_ok=True)
    subprocess.run(["python", "main.py", str(model_id)])

    print(subprocess.run(["./main", str(model_id)])) # + list(map(str, second_ids)) + [str(model_id)]))
    model_id += 1
