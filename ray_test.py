import cma
import torch
import gym
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import pprint
from functools import partial
from cma_es import Walker_AI, eval_parameters
import ray
from ray.util import ActorPool

ray.init()

@ray.remote
class Worker(object):
    def __init__(self, duration):
        self.duration = duration
        #self.env = gym.make("BipedalWalker-v3")
        #self.agent = Walker_AI()

    def execute(self, param):
        return eval_parameters(param, self.agent, self.env, self.duration)


x0 = parameters_to_vector(Walker_AI().parameters())
opts = cma.CMAOptions()
# pprint.pprint(cma.CMAOptions().match('size'))
opts.set("maxiter", 30)

es = cma.CMAEvolutionStrategy(x0, 0.02, opts)
# gym env
duration = 250

# Create several Worker actors.
workers = [Worker.remote(duration) for _ in range(4)]
pool = ActorPool(workers)

while not es.stop():
    solutions = es.ask()
    # Execute tasks on them in parallel.
    result_ids = [workers[i].execute.remote(solutions[i]) for i in range(len(workers))]
    # Get the results
    results = ray.get(result_ids)
    print(type(results))
    es.tell(solutions, results)
    es.logger.add()  # write data to disc to be plotted
    es.disp()

agent = Walker_AI()
vector_to_parameters(torch.Tensor(es.result[0]), agent.parameters())
torch.save(agent.state_dict(), "walker_O.pth")
cma.plot()