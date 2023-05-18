import matplotlib.pyplot as plt


def multiply(v1, v2):
    """
    计算两个向量的叉积
    """
    return v1.x * v2.y - v2.x * v1.y


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, other):
        """
        重载减法运算，计算两点之差，实际上返回的表示该点到另一点的向量
        :param other: 另一个点
        :return: 该点到另一点的向量
        """
        return Point(self.x - other.x, self.y - other.y)


class Segment:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def straddle(self, another_segment):
        """
        :param another_segment: 另一条线段
        :return: 如果另一条线段跨立该线段，返回True；否则返回False
        """
        v1 = another_segment.point1 - self.point1
        v2 = another_segment.point2 - self.point1
        vm = self.point2 - self.point1
        if multiply(v1, vm) * multiply(v2, vm) <= 0:
            return True
        else:
            return False

    def is_cross(self, another_segment):
        """
        :param another_segment: 另一条线段
        :return: 如果两条线段相互跨立，则相交；否则不相交
        """
        if self.straddle(another_segment) and another_segment.straddle(self):
            return True
        else:
            return False

    def cross(self, another_segment):
        if self.is_cross(another_segment):
            # print('两线段相交.')
            return True
        else:
            # print('两线段不相交.')
            return False
        # plt.plot([self.point1.x, self.point2.x], [self.point1.y, self.point2.y])
        # plt.plot([another_segment.point1.x, another_segment.point2.x],
        #          [another_segment.point1.y, another_segment.point2.y])
        # plt.show()


if __name__ == '__main__':
    A = Point(0, 0)
    B = Point(2, 2)
    C = Point(1, 1)
    D = Point(0, 2)
    AB = Segment(A, B)
    CD = Segment(C, D)
    a = AB.cross(CD)
    print(a)