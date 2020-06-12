# Setup

## Dependencies
cmake is required.
### Python
with conda environment
```bash
conda create -n env python=3.8.2
pip install -r requirements.txt
```

### c++
Download tensorflow c api from - https://www.tensorflow.org/install/lang_c
and follow steps:
* setup
* extract
* linker

build c++ files
```bash
mkdir -p build
cd build
cmake ..
make
```
After that 2 executables are compiled `self_play` and `eval`


from root directory
```
./build/self_play id
```

Will load model with id=id and play 300 games using model.


from root directory
```
./build/eval id1 id2 nr_sims
```

Will load models with id1 and id2 and play one game using nr_sims simulation for each monte carlo tree evaluation.

Human can play with ai using:
from root directory
```
./build/eval id1
```
Model is then playing as 'x' using around 4200 simulations.

from root directory
```
python self_play_train.py id
```

Will start training procces from id. First model is either created if id=0 or loaded and trained on last 10 models data. Then new model with id+1 self play 300 games. This goes until stopped.


Models are excepted to be stored in `models/mini-zero-id`

Data from selfplay of model i is stored in `data/i/`

file utils.py provides some utilities to analise model.

#TODO: add example here or as jupyter notebook.
