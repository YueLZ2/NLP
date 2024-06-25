import os
import random
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
import math
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as patches


def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Q-learning类的实现
class Q_table:
    def __init__(self, length, height, actions=4, alpha=0.1, gamma=0.9):
        self.table = [0] * actions * length * height  # initialize all Q(s,a) to zero
        self.actions = actions
        self.length = length
        self.height = height
        self.alpha = alpha
        self.gamma = gamma

    def _index(self, a, x, y):
        return a * self.height * self.length + x * self.length + y

    def _epsilon(self):
        return 0.1

    def take_action(self, x, y, num_episode):
        if random.random() < self._epsilon():
            return int(random.random() * 4)
        else:
            actions_value = [self.table[self._index(a, x, y)] for a in range(self.actions)]
            return actions_value.index(max(actions_value))

    def max_q(self, x, y):
        actions_value = [self.table[self._index(a, x, y)] for a in range(self.actions)]
        return max(actions_value)

    def update(self, a, s0, s1, r, is_terminated):
        q_predict = self.table[self._index(a, s0[0], s0[1])]
        if not is_terminated:
            q_target = r + self.gamma * self.max_q(s1[0], s1[1])
        else:
            q_target = r
        self.table[self._index(a, s0[0], s0[1])] += self.alpha * (q_target - q_predict)


class Maze:
    def __init__(self, args, size=40):
        self.args = args
        self.size = size
        self.maze = np.zeros((size, size))
        self.start_pos = (0, 0)
        self.goal_area = [(i, j) for i in range(30, 40) for j in range(30, 40)]
        self.paved_area = []  # 记录走过的路径（终点内）
        self.max_goal_distance = distance((1, 1), (30, 30))  # 距离终点最长的距离
        self.goal_pos = self._reset_goal()
        self.steps = 0
        self.max_steps = 500
        self.animation_set = []
        self.entered_goal_area = False  # 标注首次进入
        # History of agent's positions
        self.position_history = []
        # Dictionary to count occurrences of positions
        self.position_counts = {}

        # Initialize obstacles and cliffs
        self._obstacles_and_cliffs()

    def _reset_goal(self):
        return random.choice(self.goal_area)

    def _obstacles_and_cliffs(self):
        obstacles = [(1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15),
                     (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25),
                     (10, 3), (11, 4), (12, 5), (13, 6), (14, 7), (15, 8), (16, 9), (17, 10), (18, 11), (19, 12),
                     (20, 13),
                     (23, 4), (24, 4), (25, 4), (26, 4), (27, 4), (28, 4), (29, 4),
                     (30, 4), (31, 4), (32, 4), (33, 4), (34, 4), (35, 4), (36, 4),
                     (25, 22), (25, 23), (25, 24), (25, 25), (25, 26), (25, 27), (25, 28), (25, 29), (25, 30),
                     (25, 31), (25, 32), (25, 33), (25, 34), (25, 35), (25, 36), (25, 37), (25, 38), (25, 39), ]
        cliffs = [(2, 0), (2, 1),
                  (22, 2), (22, 3), (22, 4), (22, 5), (22, 6), (22, 7), (22, 8),
                  (24, 15), (24, 16), (24, 17), (24, 18), (24, 19), (24, 20), (24, 21),
                  (36, 25), (37, 25), (38, 25), (39, 25), ]

        for obs in obstacles:
            self.maze[obs] = -1  # obstacle

        for clf in cliffs:
            self.maze[clf] = -2  # cliff

    def reset(self):
        self.animation_set = []  # 动画集合重置
        self.paved_area = []  # 走过路径重置
        self.entered_goal_area = False  # 是否进入终点区域重置
        self.agent_pos = self.start_pos
        self.steps = 0
        self.goal_pos = self._reset_goal()
        self.position_history = []
        self.position_counts = {}
        return self.agent_pos

    def step(self, action):
        # move
        x, y = self.agent_pos
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.size - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.size - 1, y + 1)

        # Update position history
        current_position = self.agent_pos
        self.position_history.append(current_position)
        if len(self.position_history) > 100:
            self.position_history.pop(0)

        self.steps += 1
        # reward = -1
        if not self.entered_goal_area:
            reward = -5 * (distance((x, y), (30, 30)) / self.max_goal_distance)  # 距离终点越近，惩罚越小
            # reward = -1
            # elif not self.entered_goal_area and x > 20 and y > 20 and self.maze[x, y] != -1 and self.maze[x, y] != -2:
            #     reward = -5

            # Update position counts
            if current_position in self.position_counts:
                self.position_counts[current_position] += 1
            else:
                self.position_counts[current_position] = 1

            # 检查重复数
            if self.position_counts[current_position] >= 5:
                reward = -10  # 重复超过5给予更大的负奖励
        else:
            if (x, y) in self.paved_area:
                reward = -5  # 走了走过的路，小惩罚
            elif (x, y) not in self.goal_area:
                reward = -20  # 走出去了，大惩罚
            else:
                reward = -1  # 积极探索，奖励
                self.paved_area.append((x, y))

        if self.maze[x, y] == -1:
            return self.agent_pos, reward, False  # meet obstacle

        if self.maze[x, y] == -2:
            reward = -100  # 死了，给个大惩罚
            return (x, y), reward, True  # fell into cliff

        self.agent_pos = (x, y)

        if self.agent_pos == self.goal_pos:
            reward = 1000  # 到终点的大大奖励
            done = True
        elif self.agent_pos in self.goal_area:
            # 如果还没有进入过目标区域，给予一次奖励并设置标志
            if not self.entered_goal_area:
                reward = 500  # 阶段性大奖励
                self.entered_goal_area = True
            # reward = 5
            done = False
        elif self.steps >= self.max_steps:
            done = True
        else:
            done = False

        if self.steps % 20 == 0:
            self.goal_pos = self._reset_goal()
            self.paved_area = []

        return self.agent_pos, reward, done

    def show(self):
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)

        for i in range(self.size):
            for j in range(self.size):
                if self.maze[i, j] == -1:
                    rect = patches.Rectangle((j, self.size - i - 1), 1, 1,
                                             linewidth=1, edgecolor='black', facecolor='grey')
                    ax.add_patch(rect)
                elif self.maze[i, j] == -2:
                    rect = patches.Rectangle((j, self.size - i - 1), 1, 1,
                                             linewidth=1, edgecolor='black', facecolor='red')
                    ax.add_patch(rect)

        # start position
        rect = patches.Rectangle((self.start_pos[1], self.size - self.start_pos[0] - 1), 1, 1,
                                 linewidth=1, edgecolor='black', facecolor='yellow')
        ax.add_patch(rect)

        # goal area & goal
        for pos in self.goal_area:
            if pos == self.goal_pos:
                rect = patches.Rectangle((pos[1], self.size - pos[0] - 1), 1, 1,
                                         linewidth=1, edgecolor='black',
                                         facecolor='green')
            else:
                rect = patches.Rectangle((pos[1], self.size - pos[0] - 1), 1, 1,
                                         linewidth=1, edgecolor='black',
                                         facecolor='orange')
            ax.add_patch(rect)

        # agent position
        rect = patches.Rectangle((self.agent_pos[1], self.size - self.agent_pos[0] - 1), 1, 1,
                                 linewidth=1, edgecolor='black', facecolor='blue')
        ax.add_patch(rect)

        fig.savefig('./temp/temp.png')
        image = cv2.imread('./temp/temp.png')
        self.animation_set.append(image)
        plt.close(fig)

    def show_animation(self, name):
        # 保存动画（需要在过程中持续调用）
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(os.path.join(self.args.save_dir, name), fourcc, self.args.fps, (1000, 1000))
        for img in self.animation_set:
            video.write(img)


# 使用Q-learning在自定义迷宫中行走
def maze_walk():
    args = type('', (), {})()
    args.save_dir = './temp'
    args.fps = 30
    env = Maze(args, size=40)
    table = Q_table(length=40, height=40)
    for num_episode in range(4000):
        # within the whole learning process
        episodic_reward = 0
        is_terminated = False
        s0 = env.reset()
        while not is_terminated:
            action = table.take_action(s0[0], s0[1], num_episode)
            s1, r, is_terminated = env.step(action)
            table.update(action, s0, s1, r, is_terminated)
            episodic_reward += r
            if env.steps % 1 == 0 and num_episode >= 3000:
                env.show()
            s0 = s1
        if episodic_reward >= 500 and num_episode >= 3000:
            print("Episode: {}, Score: {}".format(num_episode, episodic_reward))
            env.show_animation('maze_walk.avi')
            break
        if num_episode % 1 == 0:
            print("Episode: {}, Score: {}".format(num_episode, episodic_reward))
        env.reset()


maze_walk()
