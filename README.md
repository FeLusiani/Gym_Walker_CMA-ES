# Bipedal Walker trained with CMA-ES

This is a project I'm currently working on.

A feed-forward neural network is trained to pilot a [Bipedal Walker](https://gym.openai.com/envs/BipedalWalker-v2/) (one of OpenAI Gym environments) using a [CMA Evolutionary Strategy](https://en.wikipedia.org/wiki/CMA-ES) (using [pycma](https://github.com/CMA-ES/pycma) package). 

## Setup

To install Gym (and the box2 environments) from source:

- Install gym
```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

- Install the box2d environments
```
pip install -e .[box2d]
```

## Usage

Use ```run_training.py``` to train and save the model.
```
$ python ./Gym_Walker_CMA-ES/run_training.py -h
usage: run_training.py [-h] [--duration DURATION] [--n_gens N_GENS]
                       [--std STD] [--file FILE] [--no_multiproc] [--logging]

Train model with cma-es

optional arguments:
  -h, --help           show this help message and exit
  --duration DURATION  duration of episode
  --n_gens N_GENS      n of generations
  --std STD            starting std
  --file FILE          file to save model
  --no_multiproc       disable multiprocessing
  --logging            disable multiprocessing
```

Use ```test.py``` to see the model in action.
```
$ python ./Gym_Walker_CMA-ES/test.py -h
usage: test.py [-h] [--duration DURATION] model_file

Test model

positional arguments:
  model_file           file to load as model

optional arguments:
  -h, --help           show this help message and exit
  --duration DURATION  duration of episode
```