import ex4.Planning as Qu
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from random import random


def CalcAlpha(count, calc_type):
    if calc_type == 0:
        return 1 / count
    elif calc_type == 1:
        return 0.01
    else:
        return 10 / (100 + count)


def TD0(queue_model, alpha_type, iteration_num):
    V = np.zeros(queue_model.policy_len)
    curr_state_idx = queue_model.policy_len - 1
    c_max_policy = queue.buildCMaxPolicy('cost')
    V_c_max, _ = queue.calcValueFunction(c_max_policy, 0.999)

    inf_norm = []
    s0_abs = []
    valid_states = [31]
    counter = np.ones(len(c_max_policy))

    for i in range(iteration_num):
        if curr_state_idx == 0:
            curr_state_idx = 31
            valid_states = [31]
        action = c_max_policy[curr_state_idx] - 1
        r, next_state_idx = queue.Simulate(curr_state_idx, action)
        dn = r + V[next_state_idx] - V[curr_state_idx]
        an = CalcAlpha(counter[curr_state_idx], alpha_type)

        V[curr_state_idx] += (an * dn)

        if next_state_idx != curr_state_idx:
            valid_states.append(next_state_idx)
        else:
            counter[curr_state_idx] += 1

        curr_state_idx = next_state_idx
        diff_vec = abs(V - V_c_max)

        inf_norm.append(max(diff_vec[valid_states]))
        s0_abs.append(diff_vec[31])
    return inf_norm, s0_abs


def TDLambda(queue_model, alpha_type, iteration_num, lamda):
    V = np.zeros(queue_model.policy_len)
    e = np.zeros(queue_model.policy_len)
    curr_state_idx = queue_model.policy_len - 1
    c_max_policy = queue.buildCMaxPolicy('cost')
    V_c_max, _ = queue.calcValueFunction(c_max_policy, 0.999)

    inf_norm = []
    s0_abs = []
    valid_states = [31]
    counter = np.ones(len(c_max_policy))

    for i in range(iteration_num):
        if curr_state_idx == 0:
            curr_state_idx = 31
            valid_states = [31]
        action = c_max_policy[curr_state_idx] - 1
        r, next_state_idx = queue.Simulate(curr_state_idx, action)
        dn = r + V[next_state_idx] - V[curr_state_idx]
        an = CalcAlpha(counter[curr_state_idx], alpha_type)
        e *= lamda
        e[curr_state_idx] += 1
        V += (an * dn * e)

        if next_state_idx != curr_state_idx:
            valid_states.append(next_state_idx)
        else:
            counter[curr_state_idx] += 1

        curr_state_idx = next_state_idx
        diff_vec = abs(V - V_c_max)

        inf_norm.append(max(diff_vec[valid_states]))
        s0_abs.append(diff_vec[31])
    return inf_norm, s0_abs


def chooseAction(epsilone, curr_state_idx, queue, Q):
    possible_jobs = queue.states_dict[curr_state_idx].jobs
    if random() < epsilone:
        action = np.argmax(Q[curr_state_idx, possible_jobs])
    else:
        action = possible_jobs[randint(len(possible_jobs) - 1)]

    return action


def QLearning(alpha_type, gamma, epsilone, iteration_num):
    Q = np.zeros((32, 5))
    curr_state_idx = 31
    counter = np.ones(32)

    for _ in range(iteration_num):
        action = chooseAction(epsilone, curr_state_idx, queue, Q)
        r, next_state_idx = queue.Simulate(curr_state_idx, action)
        an = CalcAlpha(counter[curr_state_idx], alpha_type)

        possible_jobs = queue.states_dict[next_state_idx].jobs
        addition = r + gamma * (np.max(Q[next_state_idx, possible_jobs]) - Q[curr_state_idx, action])
        Q[curr_state_idx, action] += an * addition


if __name__ == '__main__':
    mu = np.asarray([0.6, 0.5, 0.3, 0.7, 0.1])
    cost = np.asarray([1, 4, 6, 2, 9])

    queue = Qu.Queue(cost, mu)
    for i, title in enumerate(['An = 1/ count', 'An = 0.01', 'An = 10/ count']):
        inf_norm, s0 = TDLambda(queue, i, 10000, 0.1)

        plt.figure()
        plt.plot(range(1, len(inf_norm) + 1), inf_norm, '-b', label='Inf Norm')
        plt.hold(True)
        plt.plot(range(1, len(inf_norm) + 1), s0, '-g', label='S0')
        plt.legend(loc='upper right')
        plt.title(title)
        plt.show()

    print('All Done :)')
