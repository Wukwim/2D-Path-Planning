import heapq
import math
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import plotting, env, twocross


MIN_D = env.STEP * 1.42  # MIN_D必须大于STEP


class AStar:
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal_real = s_goal
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.Env = env.Env()  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles
        self.obs_circle = self.Env.obs_circle
        self.obs_rectangle = self.Env.obs_rectangle
        self.obs_boundary = self.Env.obs_boundary
        self.obs_line = self.Env.obs_line
        self.lines = self.get_lines()

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            # if s == self.s_goal:  # stop condition
            #     break
            # 修改步长范围，可以自定义
            if np.linalg.norm(np.array(s) - np.array(self.s_goal)) <= MIN_D:
                self.s_goal = s
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 0.6:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN, (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            # if s == s_goal:
            #     break
            # 修改步长范围，可以自定义了
            if np.linalg.norm(np.array(s) - np.array(self.s_goal)) <= MIN_D:
                self.s_goal = s
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def get_lines(self):
        lines = []
        for i in range(np.array(self.obs_line).shape[0]):
            K = twocross.Point(self.obs_line[i][0], self.obs_line[i][1])
            L = twocross.Point(self.obs_line[i][2], self.obs_line[i][3])
            KL = twocross.Segment(K, L)
            lines.append(KL)
        return lines

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """
        # print(np.array(self.obs_line).shape[0])  # (2,4) 判断直线碰撞
        a = twocross.Point(s_start[0], s_start[1])
        b = twocross.Point(s_end[0], s_end[1])
        ab = twocross.Segment(a, b)
        for ij in self.lines:
            if ab.cross(ij):
                return True

        # print(np.array(self.obs_circle).shape[0])  # (3,4) 判断圆形碰撞
        for j in range(np.array(self.obs_circle).shape[0]):
            if math.sqrt((s_end[0] - self.obs_circle[j][0]) ** 2 +
                         (s_end[1] - self.obs_circle[j][1]) ** 2) <= self.obs_circle[j][2]:
                return True

        # 判断边界碰撞
        if s_end[0] <= -1 or s_end[0] >= 101 or s_end[1] <= -1 or s_end[1] >= 101:
            return True

        if s_start in self.obs or s_end in self.obs:
            return True

        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal_real, self.s_goal]
        s = self.s_goal
        # print(PARENT)
        while True:
            # print(s, path)
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                path.append(self.s_start)
                break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


def cal_distance(path):
    # del path[-1]
    # print(path)
    f = open('road.txt', 'w')
    length = len(path)
    distance = 0
    max_distance = 0
    min_distance = env.STEP
    for i in range(length-1):
        # f.writelines(['x:', str(path[i][0]), ' y:', str(path[i][1]), '\n'])
        f.writelines(['x: %.3f' % path[i][0], ' y: %.3f' % path[i][1], '\n'])
        distance_new = np.linalg.norm(np.array(path[i]) - np.array(path[i + 1]))
        distance += distance_new
        max_distance = max(max_distance, distance_new)
        if distance_new > 0:
            min_distance = min(env.STEP, distance_new)

    print('寻找的节点数为 %d 个，' % (length-1), '总距离为 %.2f 米。' % distance)
    print('两点间最大距离为 %.3f 米，' % max_distance, '最小距离为 %.3f 米。' % min_distance)


def plot_fig(path, visited, start, goal, flag):
    # print(np.array(path).shape[0])
    if flag == 'a':
        path = [path]
        visited = [visited]
        row = np.array(path).shape[0]
    else:
        row = np.array(path).shape[0]
    plot = plotting.Plotting(start, goal)
    for i in range(row):
        cal_distance(path[i])
        plt.ion()
        # print(np.array(path).shape, np.array(visited).shape)
        plot.animation(path[i], visited[i], "A*")  # animation
        if i < row - 1:
            plt.pause(1)
            plt.close()
    plt.pause(0)


def main(process, flag, heuristic_type):
    start_time = time.process_time()
    t0 = time.strftime("%Y-%m-%d %X", time.localtime())

    s_start = (0, 0)
    s_goal = (100, 100)

    astar = AStar(s_start, s_goal, heuristic_type)

    if flag == 'a':
        path, visited = astar.searching()
        if not process:
            visited = []
    else:
        path, visited = astar.searching_repeated_astar(1.2)
        if not process:
            visited = [[], []]

    end_time = time.process_time()
    print('开始时间:', t0, '------')
    print('结束时间:', time.strftime("%Y-%m-%d %X", time.localtime()), '------')
    print('运行时间是 %.2f 秒！' % (end_time - start_time))

    plot_fig(path, visited, s_start, s_goal, flag)


if __name__ == '__main__':
    process = True
    # process = False

    flag = 'a'
    # flag = 'a_repeated'

    # heuristic_type = "euclidean"  # 距离短，速度慢
    heuristic_type = "manhattan"
    print('H 是：%s 函数' % str(heuristic_type))

    main(process, flag, heuristic_type)
