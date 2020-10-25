import cma
import torch
import gym
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import pprint
from multiprocessing import Pool, Queue, Manager, cpu_count
from functools import partial
from model import Walker_AI, eval_agent
import argparse
from abc import ABC, abstractmethod
from pathlib import Path


# global so it can be used with multi-processing
global agent
agent = Walker_AI()
# global so it can be used with multi-processing
global env
env = gym.make("BipedalWalker-v3")


def eval_parameters(param, duration: int):
    global agent
    global env
    env.reset()
    vector_to_parameters(torch.Tensor(param), agent.parameters())
    return -eval_agent(agent, env, duration)


class Evaluator(ABC):
    @abstractmethod
    def eval(self, solutions: list) -> list:
        raise NotImplementedError


class _Parallel_evaluator(Evaluator):
    def __init__(self, duration):
        self._pool = Pool()
        self._eval_func = partial(eval_parameters, duration=duration)

    def eval(self, solutions: list) -> list:
        function_values = self._pool.map(self._eval_func, solutions)
        return function_values

    def __del__(self):
        self._pool.close()
        self._pool.join()


class _Normal_evaluator(Evaluator):
    def __init__(self, duration):
        self.duration = duration

    def eval(self, solutions: list) -> list:
        function_values = [eval_parameters(x, self.duration) for x in solutions]
        return function_values


def create_evaluator(duration: int, multiproc=True) -> Evaluator:
    if multiproc:
        return _Parallel_evaluator(duration)
    else:
        return _Normal_evaluator(duration)


def create_save_path(args) -> Path:
    """
    Generates saving path from script arguments.
    The path will be [args.dir]/[model_name], where
    [model_name] is either args.filename (if set),
    or a generated name reporting the training parameters.

    Args:
        args : arguments of the script (output of parse_args function).

    Returns:
        Path: saving path for the model
    """
    dir_path = Path(args.dir)
    dir_path.mkdir(exist_ok=True)
    filename = args.name
    if filename is None:
        filename = f"walker_D{args.duration}_N{args.n_gens}_STD{args.std:.2E}.pth"

    return Path(args.dir) / Path(filename)


parser = argparse.ArgumentParser(description="Train model with cma-es")
parser.add_argument("--duration", help="duration of episode", type=int, default=500)
parser.add_argument("--n_gens", help="n of generations", type=int, default=50)
parser.add_argument("--std", help="starting std", type=float, default=0.3)
parser.add_argument("--name", help="filename to save model")
parser.add_argument("--dir", help="dir path to save model", default=".")
parser.add_argument(
    "--no_multiproc", help="disable multiprocessing", action="store_true"
)
parser.add_argument("--logging", help="enable cma logging", action="store_true")
args = parser.parse_args()


# we don't need gradient computing
torch.set_grad_enabled(False)
# starting solution is Walker_AI() default parameters
x0 = parameters_to_vector(Walker_AI().parameters())
opts = cma.CMAOptions()
# to find other CMA options use
# pprint.pprint(cma.CMAOptions().match('[keyword]'))

# set max number of iterations (n of generations)
opts.set("maxiter", args.n_gens)

es = cma.CMAEvolutionStrategy(x0, args.std, opts)
evaluator = create_evaluator(args.duration, not args.no_multiproc)

while not es.stop():
    solutions = es.ask()
    function_values = evaluator.eval(solutions)
    es.tell(solutions, function_values)
    if args.logging:
        es.logger.add()  # write data to disc to be plotted
    es.disp()

vector_to_parameters(torch.Tensor(es.result[0]), agent.parameters())
file_path = create_save_path(args)
print(f"\n Saving model at {file_path}")
torch.save(agent.state_dict(), file_path)
# cma.plot()