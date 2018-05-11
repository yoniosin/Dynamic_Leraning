import numpy as np

P1 = np.array([[0, 1/2, 1/2], [2/3, 0, 1/3], [3/4, 1/4, 0]])
P2 = np.array([[0, 1/8, 7/8], [1/2, 0, 1/2], [1/4, 3/4, 0]])
P = {1: P1, 2: P2}

R1 = np.array([0.2, 1, 0.5])
R2 = np.array([0.7, 0, 0.5])
R = {1: R1, 2: R2}


def calcReward(a_list, prob):
    Reward = 0
    for a in a_list:
        Reward += np.dot(prob, R[a])
        prob = np.dot(prob, P[a])

    return Reward


if __name__ == '__main__':
    init_prob = np.array([1, 0, 0])
    a_list = np.array([2, 1, 2])

    print(calcReward(a_list, init_prob))




