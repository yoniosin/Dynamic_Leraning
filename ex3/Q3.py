from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

prob_vec = np.ones(10, float)
prob_vec[8] = 4
prob_vec /= 13

helper_dict = {}


def calcTerminalYProb(x, y):
    if (x, y) in helper_dict.keys():
        return helper_dict[(x, y)]

    if x <= 21 < y or 17 <= y < x <= 21:
        helper_dict[(x, y)] = 1
        return 1

    if max(16, x) < y <= 21 or x > 21:
        helper_dict[(x, y)] = -1
        return -1

    if y >= 17 and y == x:
        helper_dict[(x, y)] = 0
        return 0

    rewardVec = np.zeros(10)
    for i in range(2, 12):
        rewardVec[i-2] = calcTerminalYProb(x, y + i)
    tmp = np.sum(rewardVec * prob_vec)
    helper_dict[(x, y)] = tmp
    return helper_dict[(x, y)]


def ValIterationMDP(x, y, v_dict, p_dict):
    if (x, y) in v_dict.keys():
        return v_dict[(x, y)]

    if x >= 21:
        return calcTerminalYProb(x, y)

    hits_vec = np.zeros(10)
    for i in range(2, 12):
        hits_vec[i-2] = ValIterationMDP(x + i, y, v_dict, p_dict)

    hits = np.sum(hits_vec * prob_vec)
    sticks = calcTerminalYProb(x, y)
    v_dict[(x, y)] = max(hits, sticks)
    p_dict[(x, y)] = 1 if hits >= sticks else 0

    return v_dict[(x, y)]


if __name__ == '__main__':
    value_dict = {}
    policy_dict = {}

    value_mat = np.zeros((18, 10))
    policy_mat = np.zeros((18, 10))
    value_mat[17, :] = np.ones(10)
    for x in range(4, 21):
        for y in range(2, 12):
            value_mat[x - 4, y - 2] = ValIterationMDP(x, y, value_dict, policy_dict)
            policy_mat[x - 4, y - 2] = policy_dict[(x, y)]

    print(ValIterationMDP(10, 21, value_dict, policy_dict))
    print('All Done!')

    plt.figure()
    ax = plt.axes(projection='3d')
    x = range(4, 22)
    y = range(2, 12)
    X, Y = np.meshgrid(y, x)

    plt.colorbar(ax.plot_surface(X, Y, value_mat, rstride=1, cstride=1, cmap='viridis', edgecolor='none'))
    ax.set_title('Value Function According Initial State')
    ax.set_xlabel('Dealer Card'), ax.set_ylabel('Beginning Sum')
    plt.show()

    plt.figure()
    plt.imshow(policy_mat, cmap='gray')
    plt.yticks(range(18), range(4, 22)), plt.ylabel('Beginning Sum')
    plt.xticks(range(10), range(2, 12)), plt.xlabel('Dealer\'s Card')
    plt.show()
