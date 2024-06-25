import random
import time
import os
import logging
import random
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
import math


def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


class Maze:
    def __init__(self, args, size=40):
        self.args = args
        self.size = size
        self.maze = np.zeros((size, size))
        self.start_pos = (0, 0)  # 起点位置
        self.goal_area = [(i, j) for i in range(30, 40) for j in range(30, 40)]
        self.paved_area = []  # 记录走过的路径（终点内）
        self.max_goal_distance = distance((1, 1), (30, 30))  # 距离终点最长的距离
        self.goal_pos = self._reset_goal()  # 初始化终点位置
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
        self.guiding_points = [(2,3), (3,3) ,(4,3), (10, 4), (11, 5), (12, 6), (13, 7), (14, 8), (15, 9), (16, 10),
                               (17, 11), (18, 12), (19, 13), (20, 14), (21, 14), (22, 14), (23, 14),
                               (24, 14), (25, 14), (26, 14), (27, 14), (28, 15)]

    def _reset_goal(self):
        return random.choice(self.goal_area)

    def _obstacles_and_cliffs(self):
        obstacles = [(1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15),
                     (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25),
                     (10, 3), (11, 4), (12, 5), (13, 6), (14, 7), (15, 8), (16, 9), (17, 10), (18, 11), (19, 12),
                     (20, 13), (23, 4), (24, 4), (25, 4), (26, 4), (27, 4), (28, 4), (29, 4),
                     (30, 4), (31, 4), (32, 4), (33, 4), (34, 4), (35, 4), (36, 4),
                     (25, 22), (25, 23), (25, 24), (25, 25), (25, 26), (25, 27), (25, 28), (25, 29), (25, 30),
                     (25, 31), (25, 32), (25, 33), (25, 34), (25, 35), (25, 36), (25, 37), (25, 38), (25, 39), ]
        cliffs = [(2, 0), (2, 1),
                  (22, 2), (22, 3), (22, 4), (22, 5), (22, 6), (22, 7), (22, 8),
                  (24, 15), (24, 16), (24, 17), (24, 18), (24, 19), (24, 20), (24, 21),
                  (36, 25), (37, 25), (38, 25), (39, 25), ]

        guiding_points = [(10, 4), (11, 5), (12, 6), (13, 7), (14, 8), (15, 9), (16, 10),
                          (17, 11), (18, 12), (19, 13), (20, 14), (21, 14), (22, 14), (23, 14),
                          (24, 14), (25, 14), (26, 14), (27, 14), (28, 15)]

        for obs in obstacles:
            self.maze[obs] = -1  # obstacle

        for clf in cliffs:
            self.maze[clf] = -2  # cliff

        for gui in guiding_points:
            self.maze[gui] = -3

    def reset(self):
        self.animation_set = []  # 动画集合重置
        self.paved_area = []  # 走过路径重置
        self.entered_goal_area = False  # 是否进入终点区域重置
        self.agent_pos = self.start_pos
        self.steps = 0
        self.goal_pos = self._reset_goal()
        self.position_history = []
        self.position_counts = {}
        self.goal_reached = False  # 重置标志
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
        if not self.entered_goal_area:
            # reward = -1
            reward = -5 * (distance((x, y), (30, 30)) / self.max_goal_distance)  # 距离终点越近，惩罚越小
            if current_position in self.position_counts:
                self.position_counts[current_position] += 1
            else:
                self.position_counts[current_position] = 1

            # 检查重复数
            if self.position_counts[current_position] >= 3:
                reward = -10  # 重复超过3给予更大的负奖励
        else:
            if (x, y) in self.paved_area:
                reward = -5  # 走了走过的路，小惩罚
            elif (x, y) not in self.goal_area:
                reward = -20  # 走出去了，大惩罚
            else:
                reward = -1  # 积极探索，奖励
                self.paved_area.append((x, y))

        # 检查引导点
        if (x, y) in self.guiding_points:
            reward += 20  # Increase reward for reaching a guiding point
            self.guiding_points.remove((x, y))  # Remove the guiding point once reached

        if action == 1:         # 向下
            if self.size - 6 >= x:
                reward += 15
        elif action == 0:
            if self.size -5 >=x:
                reward += -5
        elif action == 3:
            reward += 13
        # 检查是否遇到障碍
        if self.maze[x, y] == -1:
            reward += -20
            return self.agent_pos, reward, False  # meet obstacle

        # 是否掉入悬崖
        if self.maze[x, y] == -2:
            reward = -100  # 死了，给个大惩罚
            return (x, y), reward, True  # fell into cliff

        self.agent_pos = (x, y)

        if self.agent_pos == self.goal_pos:
            reward = 1000  # 到终点的大大奖励
            self.goal_reached = True  # 抵达终点
            done = True
        elif self.agent_pos in self.goal_area:
            # 如果还没有进入过目标区域，给予一次奖励并设置标志
            if not self.entered_goal_area:
                reward = 500  # 阶段性大奖励
                self.entered_goal_area = True
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
                elif self.maze[i, j] == -3:
                    rect = patches.Rectangle((j, self.size - i - 1), 1, 1,
                                             linewidth=1, edgecolor='black', facecolor='violet')
                    ax.add_patch(rect)

        rect = patches.Rectangle((self.start_pos[1], self.size - self.start_pos[0] - 1), 1, 1,
                                 linewidth=1, edgecolor='black', facecolor='yellow')
        ax.add_patch(rect)
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
        rect = patches.Rectangle((self.agent_pos[1], self.size - self.agent_pos[0] - 1), 1, 1,
                                 linewidth=1, edgecolor='black', facecolor='blue')
        ax.add_patch(rect)
        fig.savefig('./temp/temp.png')
        image = cv2.imread('./temp/temp.png')
        self.animation_set.append(image)
        print(f"Added frame {len(self.animation_set)} to animation_set")  # 调试信息
        plt.close(fig)

    def show_animation(self, name):
        # 保存动画
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(os.path.join(self.args.save_dir, name), fourcc, self.args.fps, (1000, 1000))
        for img in self.animation_set:
            video.write(img)
        video.release()
