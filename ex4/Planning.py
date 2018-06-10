import numpy as np
import matplotlib.pyplot as plt
mu = np.asarray([0.6, 0.5, 0.3, 0.7, 0.1])
cost = np.asarray([1, 4, 6, 2, 9])
mc = mu * cost


def buildCMaxPolicy():
    policy = np.zeros(2 ** len(cost), dtype=int)
    for s in range(2 ** len(cost)):
        curr_s_bit = np.unpackbits(np.array([[s]], dtype=np.uint8), axis=1)
        option_cost = (curr_s_bit[0, -len(cost):]) * cost
        chosen_s = np.argmax(option_cost)
        policy[s] = chosen_s + 1

    return policy


def set_bit(value, bit):
    return value | (1 << bit)


def clear_bit(value, bit):
    return value & ~(1 << bit)


def calcNextState(curr_state, selected_job):
    bit_to_change = len(cost) - 1 - selected_job
    next_state = clear_bit(curr_state, bit_to_change)
    return  next_state

def calcValueFunction(policy, gamma):
    policy = policy - 1
    pi_len = len(policy)
    P = np.zeros((pi_len, pi_len))
    l = np.zeros((pi_len))

    for curr_state, selected_job in enumerate(policy):
        P[curr_state, curr_state] = 1 - mu[selected_job]
        next_state = calcNextState(curr_state, selected_job)
        P[curr_state, next_state] += mu[selected_job]
        curr_state_bit = np.unpackbits(np.array([[curr_state]], dtype=np.uint8), axis=1)
        l[curr_state] += sum(curr_state_bit[0, -len(cost):]*cost)

    V = np.dot(np.linalg.inv(np.eye(pi_len)-gamma * P), l)
    return V, P


def policyIteration(policy, gamma):
    policy = policy - 1
    nextV, P = calcValueFunction(policy, gamma)
    conToNextIter = True
    while conToNextIter:
        prevV = nextV
        nextV = np.zeros(len(nextV))
        nextPolicy = np.zeros(len(policy))
        for s, curr_state in enumerate(policy):
            curr_s_bit = np.unpackbits(np.array([[curr_state]], dtype=np.uint8), axis=1)[0,-len(cost)]
            possible_jobs = np.where(curr_s_bit == 1)
            v_for_curr_state = np.zeros(len(possible_jobs))
            for a_idx, action in enumerate(possible_jobs):
                next_state = calcNextState(s, action)
                v_for_curr_state[a_idx] = gamma * P[next_state, :] * V
            nextV[s] = np.max(v_for_curr_state)
            nextPolicy[s] = possible_jobs[np.argmax(v_for_curr_state)]
        conToNextIter = np.equal(prevV, nextV)

    return nextPolicy




if __name__ == '__main__':
    c_max_policy = buildCMaxPolicy()
    V_c_max = calcValueFunction(c_max_policy, 0.9)

    plt.figure()
    plt.stem(range(1, 32), c_max_policy[1:])
    plt.xlabel('state'), plt.ylabel('action')
    plt.show()

    policyIteration(c_max_policy, 0.9)
    print('all done')