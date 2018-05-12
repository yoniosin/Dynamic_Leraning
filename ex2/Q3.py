import numpy as np
import itertools as it

P1 = np.array([[0, 1/2, 1/2], [2/3, 0, 1/3], [3/4, 1/4, 0]])
P2 = np.array([[0, 1/8, 7/8], [1/2, 0, 1/2], [3/4, 1/4, 0]])
P = {1: P1, 2: P2}

R1 = np.array([0.2, 1, 0.5])
R2 = np.array([0.7, 0, 0.5])
R = {1: R1, 2: R2}

stateNum = 3


def calcReward(a_list, prob):
    Reward = 0
    for a in a_list:
        Reward += np.dot(prob, R[a])
        prob = np.dot(prob, P[a])

    return Reward


def calcRewardWrapper(path_list, init_prob):
    totReward = []
    for path in path_list:
        totReward.append(calcReward(path, init_prob))

    return sum(totReward) / len(path_list)


def maxVal(a, b):
    if a > b:
        return a, 1
    else:
        return b, 2


def bestPath(steps):
    V = np.empty((3, 3))
    policy = np.empty((3, 3), dtype=int)
    vReward = np.array([[R1], [R2]])
    V[0, :] = vReward.max(0)
    policy[0, :] = (vReward.argmax(0) + 1)
    for j in range(1, steps):
        vReward = np.array([R1 + np.dot(P1, V[j-1, :]), R2 + np.dot(P2, V[j-1, :])])
        V[j, :] = vReward.max(0)
        policy[j, :] = vReward.argmax(0) + 1

    return policy, V[2, :]


if __name__ == '__main__':
    init_prob = np.array([1, 0, 0])
    a_list = np.array([2, 1, 2])

    print('a. first reward: ' + "%.2f" % calcReward(a_list, init_prob))

    path_list = list(it.product([1, 2], repeat=stateNum))
    print('b. second reward: ' + "%.2f" % calcRewardWrapper(path_list, init_prob))

    [policy, V] = bestPath(3)
    print('policy for each step is ' + '\n' + str(policy))
    print('Average reward after 3 steps starting from each state is ' + np.array_str(V, precision=2))