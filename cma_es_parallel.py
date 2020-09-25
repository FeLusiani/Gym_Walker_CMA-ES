import cma
import torch
import gym
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import pprint
from multiprocessing import Pool, Queue, Manager, cpu_count
from functools import partial
from cma_es import Walker_AI, eval_parameters


def evaluation_process(param, resource_q: Queue, duration: int):
    resource = resource_q.get()
    agent, env = resource
    value = eval_parameters(param, agent, env, duration)
    resource_q.put(resource)
    return value


x0 = parameters_to_vector(Walker_AI().parameters())
opts = cma.CMAOptions()
# pprint.pprint(cma.CMAOptions().match('size'))
opts.set("maxiter", 30)

es = cma.CMAEvolutionStrategy(x0, 0.02, opts)
# gym env
duration = 250


N_WORKERS = cpu_count()
mp_manager = Manager()
resource_queue = mp_manager.Queue()

for _ in range(N_WORKERS):
    env = gym.make("BipedalWalker-v3")
    agent = Walker_AI()
    resource_queue.put((agent, env))

evaluating_func = partial(
    evaluation_process, resource_q=resource_queue, duration=duration
)

while not es.stop():
    solutions = es.ask()
    # the non-multiprocessing equivalent would be
    # func = lambda x : eval_parameters(x, agent, env, duration)
    # function_values = [func(x) for x in solutions]
    with Pool(N_WORKERS) as p:
        function_values = p.map(evaluating_func, solutions)
    es.tell(solutions, function_values)
    es.logger.add()  # write data to disc to be plotted
    es.disp()

vector_to_parameters(torch.Tensor(es.result[0]), agent.parameters())
torch.save(agent.state_dict(), "walker_0.pth")
cma.plot()