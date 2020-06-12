import pathlib
import subprocess
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('model_id')
args = parser.parse_args()

model_id = int(args.model_id)


while True:

    pathlib.Path(f'./data/{model_id}').mkdir(parents=True, exist_ok=True)
    subprocess.run(["python", "main.py", str(model_id)])

    print(subprocess.run(["./build/self_play", str(model_id)]))
    model_id += 1
