import cma
from train import eval_agent, Walker_AI
import torch
import gym
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import pprint


def eval_parameters(param, agent, env, duration: int):
    vector_to_parameters(torch.Tensor(param), agent.parameters())
    return -eval_agent(agent, env, duration)


if __name__ == "__main__":
    agent = Walker_AI()
    x0 = parameters_to_vector(agent.parameters())
    opts = cma.CMAOptions()
    # pprint.pprint(cma.CMAOptions().match('size'))
    opts.set('maxiter', 50)

    es = cma.CMAEvolutionStrategy(x0, 0.02, opts)
    # gym env
    env = gym.make("BipedalWalker-v3")
    duration = 500

    eval_func = lambda x : eval_parameters(x, agent, env, duration)

    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [eval_func(x) for x in solutions])
        es.logger.add()  # write data to disc to be plotted
        es.disp()

    vector_to_parameters(torch.Tensor(es.result[0]), agent.parameters())
    torch.save(agent.state_dict(), 'walker_X2.pth')
    cma.plot()