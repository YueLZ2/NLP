import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import time
import os
import logging
import MAZE_r

# 经验回放中的单个步骤
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Reply Buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# DQN代理

class Agent:
    def __init__(self, state_dim, action_dim, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.args = args
        self.policy_net = DQN(state_dim, action_dim).to(args.device)
        self.target_net = DQN(state_dim, action_dim).to(args.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.memory = deque(maxlen=args.memory_size)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.update_target_every = args.update_target_every
        self.steps_done = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].item()

    def store_transition(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    # def optimize_model(self):
    #     if len(self.memory) < self.batch_size:
    #         return
    #
    #     batch = random.sample(self.memory, self.batch_size)
    #     states, actions, next_states, rewards, dones = zip(*batch)
    #
    #     states = torch.tensor(states, device=self.args.device, dtype=torch.float32)
    #     actions = torch.tensor(actions, device=self.args.device).unsqueeze(1)
    #     next_states = torch.tensor(next_states, device=self.args.device, dtype=torch.float32)
    #     rewards = torch.tensor(rewards, device=self.args.device, dtype=torch.float32)
    #     dones = torch.tensor(dones, device=self.args.device, dtype=torch.float32)
    #
    #     current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)
    #     next_q_values = self.target_net(next_states).max(1)[0].detach()
    #     expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
    #
    #     loss = F.mse_loss(current_q_values, expected_q_values)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     if self.steps_done % self.update_target_every == 0:
    #         self.target_net.load_state_dict(self.policy_net.state_dict())
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        states = torch.tensor(states, device=self.args.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.args.device).unsqueeze(1)
        next_states = torch.tensor(next_states, device=self.args.device, dtype=torch.float32)
        rewards = torch.tensor(rewards, device=self.args.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.args.device, dtype=torch.float32)

        # 使用在线网络选择下一个状态的最优动作
        next_state_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        # 使用目标网络评估该动作的Q值
        next_state_q_values = self.target_net(next_states).gather(1, next_state_actions).squeeze(1)
        # 计算期望的Q值
        expected_q_values = rewards + self.gamma * next_state_q_values * (1 - dones)

        # 当前网络对当前状态和动作的Q值
        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # 计算损失
        loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))


class Args:
    args = type('', (), {})()
    fps = 30
    num_episodes = 8000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-3
    memory_size = 10000
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    # epsilon_end = 0.01
    # epsilon_decay = 0.995
    target_update = 10
    update_target_every = 1000
    save_dir = "./model_w/DDQA_1"
    save_every = num_episodes - 1


def train(args):
    env = MAZE_r.Maze(args)
    agent = Agent(2, 4, args)
    successful_episodes = 0  # 记录成功找到终点的次数

    if not os.path.exists('./temp'):
        os.makedirs('./temp')

    for episode in range(args.num_episodes):
        state = env.reset()
        state_tensor = torch.tensor([state], device=args.device, dtype=torch.float32)
        total_reward = 0

        for t in range(env.max_steps):
            action = agent.select_action(state_tensor)
            next_state, reward, done = env.step(action)
            next_state_tensor = torch.tensor([next_state], device=args.device, dtype=torch.float32)

            agent.store_transition(state, action, next_state, reward, done)
            agent.optimize_model()
            total_reward += reward

            # # 每隔一定步数调用一次 show 方法
            # if env.steps % 3 == 0 and episode >= 20000:
            #     env.show()

            if done:
                break

            state = next_state
            state_tensor = next_state_tensor
            # Log the episode details
        logging.info(f"Episode {episode} finished with reward {total_reward}")
        print(f"Episode {episode} finished with reward {total_reward}")

        # Check if the agent found the goal and the reward is greater than 500
        if total_reward >= 0:
            successful_episodes += 1
            print(f"Episode: {episode}, Score: {total_reward}")
            # env.show_animation(f'maze_walk_{episode}.avi')
        if episode % 1 == 0:
            print(f"Episode: {episode}, Score: {total_reward}")
        if episode % args.save_every == 0:
            agent.save_model(os.path.join(args.save_dir, f'model_{episode}.pth'))
        env.reset()

    print(f"Total successful episodes: {successful_episodes}")


def test(args, agent, env, model_path):
    agent.load_model(model_path)
    env.reset()
    state = env.reset()
    state_tensor = torch.tensor([state], device=args.device, dtype=torch.float32)

    for t in range(env.max_steps):
        action = agent.select_action(state_tensor)
        next_state, reward, done = env.step(action)
        next_state_tensor = torch.tensor([next_state], device=args.device, dtype=torch.float32)

        env.show()

        if done:
            break
        state_tensor = next_state_tensor

    # Save the video os.path.join(args.save_dir, f'model_{episode}.pth')
    env.show_animation('test_animation.avi')
    print('Video saved as test_animation.avi')


if __name__ == "__main__":
    args = Args()

    # if not os.path.exists('./log'):
    #     os.makedirs('log')
    # # 创建temp目录
    # if not os.path.exists('./temp'):
    #     os.makedirs('./temp')
    # # Set up logging
    # log_filename = f"./log/training_log_{int(time.time())}.log"
    # logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    # train(args)

    env = MAZE_r.Maze(args)
    agent = Agent(2, 4, args)

    model_path = os.path.join(args.save_dir, f'model_{args.save_every}.pth')
    test(args, agent, env, model_path)
