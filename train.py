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