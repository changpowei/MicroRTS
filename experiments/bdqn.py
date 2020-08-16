# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
import gym_microrts
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicrortsMining10x10F9-v0",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    
    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=100000,
                         help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=1000,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=32,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--start-e', type=float, default=1.,
                        help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.02,
                        help="the ending epsilon for exploration")
    parser.add_argument('--exploration-fraction', type=float, default=0.10,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument('--learning-starts', type=int, default=10000,
                        help="timestep to start learning")
    parser.add_argument('--train-frequency', type=int, default=4,
                        help="the frequency of training")
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.int32,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))

class MicroRTSStatsRecorder(gym.Wrapper):

    def reset(self, **kwargs):
        observation = super(MicroRTSStatsRecorder, self).reset(**kwargs)
        self.raw_rewards = []
        return observation

    def step(self, action):
        observation, reward, done, info = super(MicroRTSStatsRecorder, self).step(action)
        self.raw_rewards += [info["raw_rewards"]]
        if done:
            raw_rewards = np.array(self.raw_rewards).sum(0)
            raw_names = [str(rf) for rf in self.rfs]
            info['microrts_stats'] = dict(zip(raw_names, raw_rewards))
            self.raw_rewards = []
        return observation, reward, done, info

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env = gym.make(args.gym_id)
env = ImageToPyTorch(env)
env = gym.wrappers.RecordEpisodeStatistics(env)
env = MicroRTSStatsRecorder(env)
env.seed(args.seed)
env.action_space.seed(args.seed)        #[100, 6, 4, 4, 4, 4, 7, 100]
env.observation_space.seed(args.seed)   #Box(27, 10, 10)
# respect the default timelimit
assert isinstance(env.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst, action_mask_lst = [], [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask, action_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)
            action_mask_lst.append(action_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst), np.array(action_mask_lst)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32*3*3, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, env.action_space.nvec.sum()), std=0.01))

    def forward(self, x):
        x = torch.Tensor(x).to(device)
        return self.network(x)

    def get_action(self, x, action=None, invalid_action_masks=None):
        logits = self.forward(x)
        split_logits = torch.split(logits, env.action_space.nvec.tolist(), dim=1)
        
        if invalid_action_masks is not None:
            split_invalid_action_masks = torch.split(invalid_action_masks, env.action_space.nvec.tolist(), dim=1)
            iams = [iam.type(torch.BoolTensor).to(device) for iam in split_invalid_action_masks]
            multi_categoricals = [torch.where(iam, logits, torch.tensor(-1e+8).to(device))
                                  for (logits, iam) in zip(split_logits, iams)]
        else:
            multi_categoricals = split_logits
        
        if action is None:
            action = torch.stack([categorical.argmax(1) for categorical in multi_categoricals])
        return action, multi_categoricals

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

rb = ReplayBuffer(args.buffer_size)
q_network = QNetwork().to(device)
target_network = QNetwork().to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
loss_fn = nn.MSELoss()
print(device.__repr__())
print(q_network)

# TRY NOT TO MODIFY: start the game
obs = env.reset()
episode_reward = 0
for global_step in range(args.total_timesteps):
    env.render()
    # ALGO LOGIC: put action logic here
    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)
    """
    invalid_action_masks為執行動作的mask，shape = [1, 229]
    [0:99]為地圖上自己單位的地方，自己單位處值為1，其餘為0
    [129:228]為地圖上敵方單位的地方，敵方單位為1，其餘為0
    其餘[100:105]:執行動作, [106:109]:移動方向, [110:113]:挖資源方向, [114:117]:放資源方向, [118:121]:建造物件方向, [122:128]:建造物件單位
    全都為1
    """
    invalid_action_masks = torch.Tensor(np.array(env.action_mask)).unsqueeze(0)
    if random.random() < epsilon:
        # action = env.action_space.sample()
        """
        將[1, 229]的Tensor根據[100, 6, 4, 4, 4, 4, 7, 100]切成8個區塊
        """
        split_invalid_action_masks = torch.split(invalid_action_masks, env.action_space.nvec.tolist(), dim=1)
        multi_categoricals = [Categorical(logits=logits) for logits in split_invalid_action_masks]
        action = torch.stack([categorical.sample() for categorical in multi_categoricals]).flatten().tolist()
    else:
        action, valid_logits = q_network.get_action(obs.reshape((1,)+obs.shape), invalid_action_masks=invalid_action_masks)
        action = action.view(-1).tolist()
        

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, reward, done, _ = env.step(action)
    episode_reward += reward

    # ALGO LOGIC: training.
    # refresh the mask for next state
    invalid_action_masks = torch.Tensor(np.array(env.action_mask)).unsqueeze(0)
    rb.put((obs, action, reward, next_obs, done, invalid_action_masks[0].tolist()))
    if global_step > args.learning_starts and global_step % args.train_frequency == 0:
        s_obs, s_actions, s_rewards, s_next_obses, s_dones, s_invalid_action_masks = rb.sample(args.batch_size)
        with torch.no_grad():
            _, valid_logits = target_network.get_action(s_next_obses, invalid_action_masks=torch.Tensor(s_invalid_action_masks).to(device))
            target_max = torch.stack([l.max(1)[0] for l in valid_logits])
            td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))
        _, old_valid_logits = q_network.get_action(s_obs)
        old_val = torch.cat([l.gather(1, a.view(-1,1)) for (a, l) in 
             zip(torch.LongTensor(s_actions.T).to(device), old_valid_logits)],1).T
        loss = ((old_val - td_target)**2).sum(1).sum()

        if global_step % 100 == 0:
            writer.add_scalar("losses/td_loss", loss, global_step)
            print(valid_logits[1][0])

        # optimize the midel
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
        optimizer.step()

        # update the target network
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook 
    obs = next_obs

    if done:
        print(list(q_network.parameters())[0].sum())
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        print(f"global_step={global_step}, episode_reward={episode_reward}")
        writer.add_scalar("charts/episode_reward", episode_reward, global_step)
        writer.add_scalar("charts/epsilon", epsilon, global_step)
        obs, episode_reward = env.reset(), 0

env.close()
writer.close()
