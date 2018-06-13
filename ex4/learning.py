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
    return np.asarray(inf_norm), np.asarray(s0_abs)


def chooseAction(epsilone, curr_state_idx, queue_model, Q):
    possible_jobs = queue_model.states_dict[curr_state_idx].jobs
    best_action = possible_jobs[np.argmin(Q[curr_state_idx, possible_jobs])]
    rand_action = randomChoice(possible_jobs)
    chosen_action = rand_action if random() < epsilone else best_action

    return chosen_action, best_action


def QLearning(queue_model, alpha_type, gamma, epsilone, iteration_num):
    Q = np.zeros((32, 5))
    curr_state_idx = 31
    counter = np.zeros(32)
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

        counter[curr_state_idx] += 1
        chosen_action, best_action = chooseAction(epsilone, curr_state_idx, queue_model, Q)
        q_policy[curr_state_idx] = best_action + 1
        r, next_state_idx = queue_model.Simulate(curr_state_idx, chosen_action)
        an = CalcAlpha(counter[curr_state_idx], alpha_type)

        possible_jobs = queue_model.states_dict[curr_state_idx].jobs
        addition = an * (r + gamma * (np.max(Q[next_state_idx, possible_jobs]) - Q[curr_state_idx, chosen_action]))
        Q[curr_state_idx, chosen_action] += addition

        if next_state_idx != curr_state_idx:
            valid_states.append(next_state_idx)

        curr_state_idx = next_state_idx

        V, _ = queue_model.calcValueFunction(q_policy, gamma)

        diff_vec = abs(V - V_mc)
        inf_norm.append(max(diff_vec[valid_states]))

        s0_abs.append(abs(V_mc[31] - np.min(Q[31, :])))

    return np.asarray(inf_norm), np.asarray(s0_abs)


def sectionG(queue_model, lamda, title_vec):
    for i, title in enumerate(title_vec):
        inf_norm, s0 = TDLambda(queue_model, i, 10000, lamda)

        plt.figure()
        plt.plot(range(1, len(inf_norm) + 1), inf_norm, '-b', label='Inf Norm')
        plt.hold(True)
        plt.plot(range(1, len(inf_norm) + 1), s0, '-g', label='S0')
        plt.legend(loc='upper right')
        plt.title(r"TD(" + str(lamda) + r"), " + title)
        plt.xlabel('Iteration Number'), plt.ylabel('Norm Value')
        plt.show()


def sectionH(queue_model, alpha, lambda_vec, rep_num, alg_iter_num, title):
    inf_norm = np.zeros(alg_iter_num)
    s0_norm = np.zeros(alg_iter_num)
    rep_div = 1 / rep_num

    for lamda in lambda_vec:
        for _ in range(rep_num):
            new_inf_norm, new_s_norm = TDLambda(queue_model, alpha, 10000, lamda)
            inf_norm += (new_inf_norm * rep_div)
            s0_norm += (new_s_norm * rep_div)

        plt.figure()
        plt.plot(range(alg_iter_num), inf_norm, '-b', label='Inf Norm')
        plt.hold(True)
        plt.plot(range(alg_iter_num), s0_norm, '-g', label='S0')
        plt.legend(loc='upper right')
        plt.title('TD(' + str(lamda) + ') ' + title)
        plt.xlabel('Iteration Number'), plt.ylabel('Norm Value')
        plt.show()


def sectionI(queue_model, gamma, epsilon, title_vec):
    for i, title in enumerate(title_vec):
        inf_norm, s0 = QLearning(queue_model, i, gamma, epsilon, 10000)
        plt.figure()
        plt.plot(range(100, len(inf_norm)), inf_norm[100:], '-b', label='Inf Norm')
        plt.hold(True)
        plt.plot(range(100, len(inf_norm)), s0[100:], '-g', label='S0')
        plt.legend(loc='upper right')
        plt.title('Q-Learning, ' + title + r"$, \epsilon = $" + str(epsilon))
        plt.xlabel('Iteration Number'), plt.ylabel('Norm Value')
        plt.show()


if __name__ == '__main__':
    mu = np.asarray([0.6, 0.5, 0.3, 0.7, 0.1])
    cost = np.asarray([1, 4, 6, 2, 9])
    title_vec = [r"$A_{n} = \frac{1} {count}$", r"$A_{n} = 0.01$", r"$A_{n} = \frac{10}{100 + count}$"]

    queue = Qu.Queue(cost, mu)
    sectionG(queue, 0, title_vec)
    sectionH(queue, 3, [0.1, 0.5, 0.9], 20, 10000, title_vec[2])
    sectionI(queue, 0.999, 0.1, title_vec)
    sectionI(queue, 0.999, 0.01, title_vec)

    print('All Done :)')
