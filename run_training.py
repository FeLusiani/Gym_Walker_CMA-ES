import cma
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import gym
from walker_cma.model import Walker_AI, eval_agent
from walker_cma.evaluator import create_evaluator
from walker_cma.utils import create_save_path
import argparse


agent = Walker_AI()
env = gym.make("BipedalWalker-v3")

parser = argparse.ArgumentParser(description="Train model with cma-es")
parser.add_argument("--duration", help="duration of episode", type=int, default=100)
parser.add_argument("--n_gens", help="n of generations", type=int, default=50)
parser.add_argument("--std", help="starting std", type=float, default=0.3)
parser.add_argument("--name", help="filename to save model")
parser.add_argument("--load", help="file to load as starting solution", default="")
parser.add_argument("--dir", help="dir path to save model", default="")
parser.add_argument(
    "--no_multiproc", help="disable multiprocessing", action="store_true"
)
parser.add_argument("--logging", help="enable cma logging", action="store_true")
args = parser.parse_args()


# we don't need gradient computing
torch.set_grad_enabled(False)
# starting solution
if args.load:
    state_dict = torch.load(args.load)
    agent.load_state_dict(state_dict)
x0 = parameters_to_vector(agent.parameters())
opts = cma.CMAOptions()
# to find other CMA options use
# pprint.pprint(cma.CMAOptions().match('[keyword]'))

# set max number of iterations (n of generations)
opts.set("maxiter", args.n_gens)

es = cma.CMAEvolutionStrategy(x0, args.std, opts)
evaluator = create_evaluator(args.duration, agent, env, not args.no_multiproc)

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