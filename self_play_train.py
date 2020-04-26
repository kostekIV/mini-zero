import pathlib
import subprocess
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('model_id')
args = parser.parse_args()

model_id = int(args.model_id)


subprocess.run("g++ cpp/main.cpp cpp/model.cpp cpp/game.cpp cpp/mct_search.cpp cpp/node.cpp -L/usr/local/lib -lgsl -lgslcblas -ltensorflow -o main -std=c++17 -O3".split())
while True:

    pathlib.Path(f'./data/{model_id}').mkdir(parents=True, exist_ok=True)
    subprocess.run(["python", "main.py", str(model_id)])

    print(subprocess.run(["./main", str(model_id)]))
    model_id += 1
