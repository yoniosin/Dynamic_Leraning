import numpy as np

helper_dict = {}


def calcTerminalYProb(x, y):
    if (x, y) in helper_dict.keys():
        return helper_dict[(x, y)]

    if y > 21 or 17 <= y < x:
        helper_dict[(x, y)] = 1
        return 1

    if y > max(16, x):
        helper_dict[(x, y)] = -1
        return -1

    if y >= 17 and y == x:
        helper_dict[(x, y)] = 0
        return 0

    rewardVec = np.zeros(10)
    for i in range(10):
        rewardVec[i] = calcTerminalYProb(x, y + (i + 2))
    helper_dict[(x, y)] = np.mean(rewardVec)
    return helper_dict[(x, y)]


if __name__ == '__main__':
    print(calcTerminalYProb(2, 14))
