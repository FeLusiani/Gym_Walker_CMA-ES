import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from functools import partial
from multiprocessing import Pool
from abc import ABC, abstractmethod
from walker_cma.model import Walker_AI, eval_agent


def eval_parameters(param, duration: int, agent, env):
    env.reset()
    vector_to_parameters(torch.Tensor(param), agent.parameters())
    return -eval_agent(agent, env, duration)

# the reason for a Evaluator object instead of just an evaluator function
# is to avoid the creation and destruction of a Pool for parallel computing
# at each evaluation cycle

class Evaluator(ABC):
    @abstractmethod
    def eval(self, solutions: list) -> list:
        raise NotImplementedError


class _Parallel_evaluator(Evaluator):
    def __init__(self, duration, agent, env):
        self._pool = Pool()
        self._eval_func = partial(eval_parameters, duration=duration, agent=agent, env=env)

    def eval(self, solutions: list) -> list:
        function_values = self._pool.map(self._eval_func, solutions)
        return function_values

    def __del__(self):
        self._pool.close()
        self._pool.join()


class _Normal_evaluator(Evaluator):
    def __init__(self, duration, agent, env):
        self.agent = agent
        self.env = env
        self.duration = duration

    def eval(self, solutions: list) -> list:
        function_values = [eval_parameters(x, self.duration, self.agent, self.env) for x in solutions]
        return function_values


def create_evaluator(duration: int, agent, env, multiproc=True) -> Evaluator:
    if multiproc:
        return _Parallel_evaluator(duration, agent, env)
    else:
        return _Normal_evaluator(duration, agent, env)