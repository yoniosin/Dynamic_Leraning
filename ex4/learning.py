import ex4.Planning as Qu
import numpy as np
import matplotlib.pyplot as plt
from random import choice as randomChoice
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
        action = possible_jobs[np.argmin(Q[curr_state_idx, possible_jobs])]
    else:
        action = randomChoice(possible_jobs)

    return action


def QLearning(queue_model, alpha_type, gamma, epsilone, iteration_num):
    Q = np.zeros((32, 5))
    curr_state_idx = 31
    counter = np.ones(32)
    q_policy = np.ones(32).astype(int)
    mc_max_policy = queue_model.buildCMaxPolicy('mc')
    V_mc, _ = queue_model.calcValueFunction(mc_max_policy, gamma)

    inf_norm = []
    s0_abs = []
    valid_states = [31]

    for curr_iter in range(iteration_num):
        if curr_state_idx == 0:
            curr_state_idx = 31
            valid_states = [31]

        action = chooseAction(epsilone, curr_state_idx, queue_model, Q)
        q_policy[curr_state_idx] = action + 1
        r, next_state_idx = queue_model.Simulate(curr_state_idx, action)
        an = CalcAlpha(counter[curr_state_idx], alpha_type)

        possible_jobs = queue_model.states_dict[curr_state_idx].jobs  # was next_state_idx
        addition = an * (r + gamma * (np.max(Q[next_state_idx, possible_jobs]) - Q[curr_state_idx, action]))
        Q[curr_state_idx, action] += addition

        curr_state_idx = next_state_idx

        if curr_iter % 50 == 0:
            V, _ = queue_model.calcValueFunction(q_policy, gamma)

            diff_vec = abs(V - V_mc)
            inf_norm.append(max(diff_vec[valid_states]))

            s0_abs.append(abs(V_mc[31] - np.min(Q[31, :])))

    return q_policy, inf_norm, s0_abs


if __name__ == '__main__':
    mu = np.asarray([0.6, 0.5, 0.3, 0.7, 0.1])
    cost = np.asarray([1, 4, 6, 2, 9])

    queue = Qu.Queue(cost, mu)
    # for i, title in enumerate(['An = 1/ count', 'An = 0.01', 'An = 10/ count']):
    #     inf_norm, s0 = TDLambda(queue, i, 10000, 0.1)
    #
    #     plt.figure()
    #     plt.plot(range(1, len(inf_norm) + 1), inf_norm, '-b', label='Inf Norm')
    #     plt.hold(True)
    #     plt.plot(range(1, len(inf_norm) + 1), s0, '-g', label='S0')
    #     plt.legend(loc='upper right')
    #     plt.title(title)
    #     plt.show()

    q_policy, inf_norm, s0 = QLearning(queue, 2, 0.99, 0.01, 10000)
    plt.figure()
    plt.plot(range(1, len(inf_norm)), inf_norm[1:], '-b', label='Inf Norm')
    plt.hold(True)
    plt.plot(range(1, len(inf_norm)), s0[1:], '-g', label='S0')
    plt.legend(loc='upper right')
    # plt.title(title)
    plt.show()
    print('All Done :)')
