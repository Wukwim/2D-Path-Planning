import os
import sys
import math
import heapq
import time
import numpy as np
import matplotlib.pyplot as plt


import plotting, env, twocross

STEP = 0.25

class BidirectionalAStar:
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.Env = env.Env()  # class Env

        # self.u_set = self.Env.motions  # feasible input set
        self.u_set = [(-STEP, 0), (-STEP, STEP), (0, STEP), (STEP, STEP),
                      (STEP, 0), (STEP, -STEP), (0, -STEP), (-STEP, -STEP)]
        self.obs = self.Env.obs  # position of obstacles
        self.obs_circle = self.Env.obs_circle
        self.obs_rectangle = self.Env.obs_rectangle
        self.obs_boundary = self.Env.obs_boundary
        self.obs_line = self.Env.obs_line
        self.lines = self.get_lines()

        self.OPEN_fore = []  # OPEN set for forward searching
        self.OPEN_back = []  # OPEN set for backward searching
        self.CLOSED_fore = []  # CLOSED set for forward
        self.CLOSED_back = []  # CLOSED set for backward
        self.PARENT_fore = dict()  # recorded parent for forward
        self.PARENT_back = dict()  # recorded parent for backward
        self.g_fore = dict()  # cost to come for forward
        self.g_back = dict()  # cost to come for backward

    def init(self):
        """
        initialize parameters
        """

        self.g_fore[self.s_start] = 0.0
        self.g_fore[self.s_goal] = math.inf
        self.g_back[self.s_goal] = 0.0
        self.g_back[self.s_start] = math.inf
        self.PARENT_fore[self.s_start] = self.s_start
        self.PARENT_back[self.s_goal] = self.s_goal
        heapq.heappush(self.OPEN_fore,
                       (self.f_value_fore(self.s_start), self.s_start))
        heapq.heappush(self.OPEN_back,
                       (self.f_value_back(self.s_goal), self.s_goal))

    def searching(self):
        """
        Bidirectional A*
        :return: connected path, visited order of forward, visited order of backward
        """

        self.init()
        s_meet = self.s_start

        while self.OPEN_fore and self.OPEN_back:
            # solve foreward-search
            _, s_fore = heapq.heappop(self.OPEN_fore)

            if s_fore in self.PARENT_back:
                s_meet = s_fore
                break

            self.CLOSED_fore.append(s_fore)

            for s_n in self.get_neighbor(s_fore):
                new_cost = self.g_fore[s_fore] + self.cost(s_fore, s_n)

                if s_n not in self.g_fore:
                    self.g_fore[s_n] = math.inf

                if new_cost < self.g_fore[s_n]:
                    self.g_fore[s_n] = new_cost
                    self.PARENT_fore[s_n] = s_fore
                    heapq.heappush(self.OPEN_fore,
                                   (self.f_value_fore(s_n), s_n))

            # solve backward-search
            _, s_back = heapq.heappop(self.OPEN_back)

            if s_back in self.PARENT_fore:
                s_meet = s_back
                break

            self.CLOSED_back.append(s_back)

            for s_n in self.get_neighbor(s_back):
                new_cost = self.g_back[s_back] + self.cost(s_back, s_n)

                if s_n not in self.g_back:
                    self.g_back[s_n] = math.inf

                if new_cost < self.g_back[s_n]:
                    self.g_back[s_n] = new_cost
                    self.PARENT_back[s_n] = s_back
                    heapq.heappush(self.OPEN_back,
                                   (self.f_value_back(s_n), s_n))

        return self.extract_path(s_meet), self.CLOSED_fore, self.CLOSED_back

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def extract_path(self, s_meet):
        """
        extract path from start and goal
        :param s_meet: meet point of bi-direction a*
        :return: path
        """

        # extract path for foreward part
        path_fore = [s_meet]
        s = s_meet

        while True:
            s = self.PARENT_fore[s]
            path_fore.append(s)
            if s == self.s_start:
                break

        # extract path for backward part
        path_back = []
        s = s_meet

        while True:
            s = self.PARENT_back[s]
            path_back.append(s)
            if s == self.s_goal:
                break

        return list(reversed(path_fore)) + list(path_back)

    def f_value_fore(self, s):
        """
        forward searching: f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g_fore[s] + self.h(s, self.s_goal)

    def f_value_back(self, s):
        """
        backward searching: f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g_back[s] + self.h(s, self.s_start)

    def h(self, s, goal):
        """
        Calculate heuristic value.
        :param s: current node (state)
        :param goal: goal node (state)
        :return: heuristic value
        """

        heuristic_type = self.heuristic_type

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

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


def cal_distance(path):
    # print(path)
    f = open('road.txt', 'w')
    length = len(path)
    distance = 0
    distance_old = 0
    max_distance = 0
    min_distance = env.STEP
    for i in range(length - 1):
        f.writelines(['x: %.3f' % path[i][0], ' y: %.3f' % path[i][1], '\n'])
        distance_new = np.linalg.norm(np.array(path[i]) - np.array(path[i + 1]))
        distance += distance_new
        max_distance = max(distance_old, distance_new)
        if distance_new > 0:
            min_distance = min(env.STEP, distance_new)
        distance_old = distance_new

    print('总距离为 %.2f 米。' % distance)
    print('两点间最大距离为 %.3f 米，' % max_distance, '最小距离为 %.3f 米。' % min_distance)


def main(process, heuristic_type):
    start_time = time.process_time()
    t0 = time.strftime("%Y-%m-%d %X", time.localtime())
    x_start = (0, 0)
    x_goal = (100, 100)

    bastar = BidirectionalAStar(x_start, x_goal, heuristic_type)

    plot = plotting.Plotting(x_start, x_goal)

    path, visited_fore, visited_back = bastar.searching()

    end_time = time.process_time()
    print('开始时间:', t0, '------')
    print('结束时间:', time.strftime("%Y-%m-%d %X", time.localtime()), '------')
    print('运行时间是 %.2f 秒！' % (end_time - start_time))

    cal_distance(path)
    if not process:
        visited_fore,  visited_back = [], []

    plt.ion()
    plot.animation_bi_astar(path, visited_fore, visited_back, "Bidirectional-A*")  # animation
    plt.pause(0)


if __name__ == '__main__':
    # process = True
    process = False

    heuristic_type = "manhattan"
    # heuristic_type = "euclidean"

    main(process, heuristic_type)
