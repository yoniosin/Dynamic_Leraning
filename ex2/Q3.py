import numpy as np
import itertools as it

P1 = np.array([[0, 1/2, 1/2], [2/3, 0, 1/3], [3/4, 1/4, 0]])
P2 = np.array([[0, 1/8, 7/8], [1/2, 0, 1/2], [1/4, 3/4, 0]])
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

    return sum(totReward) / len(path_list), max(totReward)


def bestPath(steps, initProb):
    V = np.zeros(stateNum)
    policy = []
    for i in range(steps):
        V[i] += max(np.dot(R[1], initProb), np.dot(R[2], initProb))

     V[i] = max(np.dot(R[1], initProb), np.dot(R[2], initProb))


if __name__ == '__main__':
    init_prob = np.array([1, 0, 0])
    a_list = np.array([2, 1, 2])

    print('a. first reward: ' + str(calcReward(a_list, init_prob)))

    path_list = list(it.product([1, 2], repeat=stateNum))
    print('b. second reward: ' + str(calcRewardWrapper(path_list, init_prob)))




