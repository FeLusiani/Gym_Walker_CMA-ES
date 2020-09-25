import gym  # open ai gym
import time
import numpy as np
import torch
import copy
from pathlib import Path
import argparse


class Walker_AI(torch.nn.Module):
    def __init__(self):
        super(Walker_AI, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(14, 25),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(25, 10),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(10, 4),
            torch.nn.Hardtanh(),
        )

        for param in self.parameters():
            param.requires_grad = False
        
        for layer in self.net:
            if type(layer) == torch.nn.Linear:
                layer.weight.data.fill_(0.0)
                layer.bias.data.fill_(0.0)

    def forward(self, x):
        out = self.net(x)
        return out


def init_rand_weights(m: Walker_AI):
    for layer in m.net:
        if type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.00)


def return_random_agents(num_agents):

    agents = []
    for _ in range(num_agents):

        agent = Walker_AI()
        init_rand_weights(agent)
        agents.append(agent)

    return agents


def eval_agent(agent, env, duration):
    status = env.reset()[:14]
    tot_reward = 0

    for _ in range(duration):
        action_t = agent(torch.Tensor(status))
        action_np = action_t.numpy()
        new_status, reward, done, _ = env.step(action_np)

        tot_reward = tot_reward + reward # - 0.0035 * (new_status[8] + new_status[13])
        status = new_status[:14]

        if done:
            break

    return tot_reward


def run_agents(agents, env, duration=250):
    reward_agents = []
    for agent in agents:
        reward_agents.append(eval_agent(agent, env, duration))

    return reward_agents


def mutate(agent):

    child_agent = copy.deepcopy(agent)

    mutation_power = (
        0.02  # hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
    )

    for param in child_agent.parameters():
        noise = torch.normal(0, mutation_power, param.data.size())
        param.data += noise

    return child_agent


def return_children(num_children, parents, sorted_parent_indexes):

    children = []

    # first take selected parents from sorted_parent_indexes and generate N-1 children
    for _ in range(len(parents)):
        selected_agent_index = sorted_parent_indexes[
            np.random.randint(len(sorted_parent_indexes))
        ]
        children.append(mutate(parents[selected_agent_index]))

    return children


def save_generation(agents: list, dir: Path, sorted_indexes: list):
    if not sorted_indexes:
        sorted_indexes = range(len(agents))

    for i in range(len(sorted_indexes)):
        filename = f"agent_file{i}.pth"
        path = dir / filename
        idx = sorted_indexes[i]
        torch.save(agents[idx].state_dict(), path)


def load_agent(dir: Path, id: int = 0) -> Walker_AI:
    file = dir / f"agent_file{id}.pth"
    state_dict = torch.load(file)
    agent = Walker_AI()
    for param in agent.parameters():
        param.requires_grad = False
    agent.load_state_dict(state_dict)

    return agent


def load_generation(dir: Path, size: int = 0) -> list:
    agents = []
    for _ in dir.iterdir():
        agents.append(load_agent(dir, i))
        size -= 1
        if size == 0:
            break

    # all saved generation loaded
    # but there are still agents to return
    if size > 0:
        children = return_children(size - i, agents, range(len(agents)))
        agents.extend(children)

    return agents


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument(
        "save_dir",
        nargs="?",
        help="path of directory to save last generation",
        default="./untitled_generation",
    )

    parser.add_argument(
        "--n_gens",
        help="n of generations to run",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--starting_gen",
        help="path of directory to use as starting generation",
        type=str,
    )

    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # disable gradients as we will not use them
    torch.set_grad_enabled(False)

    # initialize N number of agents
    num_agents = 500

    if args.starting_gen is None:
        agents = return_random_agents(num_agents)
    else:
        agents = load_generation(Path(args.starting_gen), num_agents)

    # How many top agents to consider as parents
    top_limit = 120

    # run evolution until X generations
    n_gens = args.n_gens
    print(f"n_gens = {n_gens}")

    # gym env
    env = gym.make("BipedalWalker-v3")

    for generation in range(n_gens):

        # return rewards of agents
        rewards = run_agents(agents, env)

        # sort by rewards
        sorted_parent_indexes = np.argsort(rewards)[::-1][
            :top_limit
        ]  # reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
        print("")
        print("")

        if generation == (n_gens - 1):
            save_generation(agents, save_dir, sorted_parent_indexes.tolist())

        top_rewards = []
        for i in sorted_parent_indexes:
            top_rewards.append(rewards[i])

        print(
            "Generation ",
            generation,
            " | Mean rewards: ",
            np.mean(rewards),
            " | Mean of top 5: ",
            np.mean(top_rewards[:5]),
        )

        # setup an empty list for containing children agents
        children_agents = return_children(num_agents, agents, sorted_parent_indexes)

        # kill all agents, and replace them with their children
        agents = children_agents