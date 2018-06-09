import numpy as np
import matplotlib.pyplot as plt
mu = np.asarray([0.6, 0.5, 0.3, 0.7, 0.1])
cost = np.asarray([1, 4, 6, 2, 9])
mc = mu * cost


def buildCMaxPolicy():
    policy = np.zeros(2 ** len(cost), dtype=int)
    for s in range(2 ** len(cost)):
        curr_s_bit = np.unpackbits(np.array([[s]], dtype=np.uint8), axis=1)
        chosen_s = np.argmax(curr_s_bit[0, -len(cost):] * cost)
        policy[s] = chosen_s + 1

    return policy


def set_bit(value, bit):
    return value | (1<<bit)


def clear_bit(value, bit):
    return value & ~(1<<bit)


def calcValueFunction(policy, gamma):
    policy = policy - 1
    pi_len = len(policy)
    P = np.zeros((pi_len, pi_len))
    r = np.zeros((pi_len))

    for curr_state, selected_job in enumerate(policy):
        P[curr_state, curr_state] = 1 - mu[selected_job]
        next_state = set_bit(curr_state, selected_job)
        P[curr_state, next_state] = mu[selected_job]
        curr_state_bit = np.unpackbits(np.array([[curr_state]], dtype=np.uint8), axis=1)
        r[curr_state] += sum(curr_state_bit[0, -len(cost):]*cost)

    V = np.dot(np.linalg.inv(np.eye(pi_len)-gamma * P), r)
    return V


if __name__ == '__main__':
    c_max_policy = buildCMaxPolicy()
    V_c_max = calcValueFunction(c_max_policy, 1)

    plt.figure()
    plt.stem(V_c_max)
    plt.show()
    print('all done')