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


def eval_parameters(param, agent, env, duration: int):
    vector_to_parameters(torch.Tensor(param), agent.parameters())
    return -eval_agent(agent, env, duration)


def evaluation_process(param, resource_q: Queue, duration: int):
    resource = resource_q.get()
    agent, env = resource
    value = eval_parameters(param, agent, env, duration)
    resource_q.put(resource)
    return value


class Evaluator(ABC):
    @abstractmethod
    def eval(self, solutions: list) -> list:
        raise NotImplementedError


class Parallel_evaluator(Evaluator):
    def __init__(self, duration):
        self._N_WORKERS = cpu_count()
        self._manager = Manager()
        self._r_queue = self._manager.Queue()

        for _ in range(self._N_WORKERS):
            env = gym.make("BipedalWalker-v3")
            agent = Walker_AI()
            self._r_queue.put((agent, env))

        self._evaluating_func = partial(
            evaluation_process, resource_q=self._r_queue, duration=duration
        )

    def eval(self, solutions: list) -> list:
        with Pool(self._N_WORKERS) as p:
            function_values = p.map(self._evaluating_func, solutions)
        return function_values


class Normal_evaluator(Evaluator):
    def __init__(self, duration):
        self.duration = duration
        self._env = gym.make("BipedalWalker-v3")
        self._agent = Walker_AI()
        self._evaluating_func = lambda x: eval_parameters(
            x, self._agent, self._env, self.duration
        )

    def eval(self, solutions: list) -> list:
        function_values = [self._evaluating_func(x) for x in solutions]
        return function_values


def create_evaluator(duration, multiproc=True) -> Evaluator:
    if multiproc:
        return Parallel_evaluator(duration)
    else:
        return Normal_evaluator(duration)


parser = argparse.ArgumentParser(description="Train model with cma-es")
parser.add_argument("--duration", help="duration of episode", type=int, default=500)
parser.add_argument("--n_gens", help="n of generations", type=int, default=50)
parser.add_argument("--std", help="starting std", type=float, default=0.3)
parser.add_argument("--file", help="file to save model", default="./walker_0.pth")
parser.add_argument(
    "--no_multiproc", help="disable multiprocessing", action="store_true"
)
parser.add_argument("--logging", help="disable multiprocessing", action="store_true")
args = parser.parse_args()

torch.set_grad_enabled(False)
x0 = parameters_to_vector(Walker_AI().parameters())
opts = cma.CMAOptions()
# pprint.pprint(cma.CMAOptions().match('size'))
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

agent = Walker_AI()
vector_to_parameters(torch.Tensor(es.result[0]), agent.parameters())
torch.save(agent.state_dict(), args.file)
# cma.plot()