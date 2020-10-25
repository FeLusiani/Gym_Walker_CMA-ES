import gym
import torch
import numpy
from pathlib import Path
from model import load_agent, Walker_AI
import argparse

parser = argparse.ArgumentParser(description="Test model")

parser.add_argument("model_file", help="file to load as model")
parser.add_argument("--duration", help="duration of episode", type=int, default=1000)

args = parser.parse_args()
model_file = Path(args.model_file)
duration = args.duration

# gym env
env = gym.make("BipedalWalker-v3")
# we don't need gradient computing
torch.set_grad_enabled(False)

for i in range(100):
    status = env.reset()[:14]
    state_dict = torch.load(model_file)
    agent = Walker_AI()
    agent.load_state_dict(state_dict)

    print(f"Test episode {i}")
    for _ in range(duration):
        action_t = agent(torch.Tensor(status))
        action_np = action_t.numpy()
        new_status, reward, done, _ = env.step(action_np)

        status = new_status[:14]

        env.render()
        if done:
            break
