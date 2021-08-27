import numpy as np

# STEP = 0.35  # 5 / 2**0.5 = 0.35
STEP = 1

# angle = np.pi/4
# motions = []
# for i in range(int(np.pi*2 / angle)):
#     x = STEP * np.cos(i*angle)
#     y = STEP * np.sin(i*angle)
#     motions.append((x, y))


class Env:
    def __init__(self):
        self.x_range = 102  # size of background
        self.y_range = 102
        self.motions = [(-STEP, 0), (-STEP, STEP), (0, STEP), (STEP, STEP),
                        (STEP, 0), (STEP, -STEP), (0, -STEP), (-STEP, -STEP)]
        # self.motions = motions
        self.obs = self.obs_map()
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.obs_line = self.obs_line()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range
        negtive_int = -2
        range_x = range(negtive_int, x)
        range_y = range(negtive_int, y)
        obs = set()

        # for i in range_x:
        #     obs.add((i, negtive_int))
        # for i in range_x:
        #     obs.add((i, y - 1))
        #
        # for i in range_y:
        #     obs.add((negtive_int, i))
        # for i in range_y:
        #     obs.add((x - 1, i))
        #
        # for i in range(10, 21):
        #     obs.add((i, 15))
        # for i in range(15):
        #     obs.add((20, i))
        #
        # for i in range(15, 30):
        #     obs.add((30, i))
        # for i in range(16):
        #     obs.add((40, i))

        return obs

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [-1, -1, 0.5, 102],
            [-1, 101, 102, 0.5],
            [-1, -1, 102, 0.5],
            [101, -1, 0.5, 102.5]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            # [14, 12, 8, 2],
            # [18, 22, 8, 3],
            # [26, 7, 2, 12],
            # [32, 14, 10, 2]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = [
            [50, 50, 25],
            [13, 30, 10],
            [87, 70, 10],
            # [30, 60, 25],  # add
            # [90, 40, 10],
            # [20, 15, 10],
        ]
        return obs_cir

    @staticmethod
    def obs_line():
        obs_line = [
            [85, 55, 47, 90],
            [19, 14, 73, 14],
            # [80, 20, 80, 50],  # add
            # [40, 10, 80, 20],
            # [20, 80, 90, 90],
        ]
        return obs_line
