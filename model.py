import torch
from pathlib import Path


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


def eval_agent(agent, env, duration):
    status = env.reset()[:14]
    tot_reward = 0

    for _ in range(duration):
        action_t = agent(torch.Tensor(status))
        action_np = action_t.numpy()
        new_status, reward, done, _ = env.step(action_np)

        # subtract reward for leg contact with the ground
        tot_reward = tot_reward + reward  - 0.035 * (new_status[8] + new_status[13])
        status = new_status[:14]

        if done:
            break

    return tot_reward


def load_agent(dir: Path, id: int = 0) -> Walker_AI:
    file = dir / f"agent_file{id}.pth"
    state_dict = torch.load(file)
    agent = Walker_AI()
    for param in agent.parameters():
        param.requires_grad = False
    agent.load_state_dict(state_dict)

    return agent