import gym
import torch
import numpy
from pathlib import Path
from train import load_agent, Walker_AI
import argparse

parser = argparse.ArgumentParser(description="Test model")

parser.add_argument("load_dir", help="path of directory to load")
parser.add_argument("--duration", help="duration of episode", type=int, default=1000)

args = parser.parse_args()
load_dir = Path(args.load_dir)
duration = args.duration

# gym env
env = gym.make("BipedalWalker-v3")

for i in range(100):
    status = env.reset()[:14]
    # winner = load_agent(load_dir, i)
    ##########################################################3
    state_dict = torch.load(load_dir)
    agent = Walker_AI()
    for param in agent.parameters():
        param.requires_grad = False
    agent.load_state_dict(state_dict)
    ##############################################################

    print(f"Individual {i}")
    for _ in range(duration):
        action_t = agent(torch.Tensor(status))
        action_np = action_t.numpy()
        new_status, reward, done, _ = env.step(action_np)

        status = new_status[:14]

        env.render()
        if done:
            break
