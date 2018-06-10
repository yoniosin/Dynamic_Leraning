from typing import Any, Union

import numpy as np
import matplotlib.pyplot as plt


mu = np.asarray([0.6, 0.5, 0.3, 0.7, 0.1])
cost = np.asarray([1, 4, 6, 2, 9])
policy_len = 2 ** len(cost)
mc = mu * cost


class State:

    def __init__(self):
        self.next_states = []
        self.jobs = []

    def calcLoss(self):
        self.loss = sum(cost[self.jobs])


states_dict = {i: State() for i in range(policy_len)}


def init(job_num):
    for state_idx in range(2 ** len(cost)):
        state = states_dict[state_idx]
        for job in range(job_num):
            if isJobWaiting(state_idx, job):
                next_state = calcNextState(state_idx, job)
                state.next_states.append(next_state)
                state.jobs.append(job)

        state.calcLoss()

    return states_dict


def isJobWaiting(state, job):
    if (state & (1 << job)) >> job == 1:
        return True
    return False


def calcNextState(state, job):
    if state == 0:
        return 0
    return state - 2 ** job


def buildCMaxPolicy(calc_type):
    policy = np.ones(policy_len)
    if calc_type == 'mc':
        cost_vec = mc
    else:
        cost_vec = cost
    for state in range(1, policy_len):
        curr_state = states_dict[state]
        option_cost = cost_vec[curr_state.jobs]
        chosen_s = curr_state.jobs[np.argmax(option_cost)]
        policy[state] = chosen_s + 1

    return policy.astype(int)


def calcValueFunction(policy, gamma):
    policy = policy - 1
    pi_len = len(policy)
    P = np.zeros((pi_len, pi_len))
    l = np.zeros(pi_len)

    for curr_state, selected_job in enumerate(policy):
        P[curr_state, curr_state] = 1 - mu[selected_job]
        next_state = calcNextState(curr_state, selected_job)
        P[curr_state, next_state] += mu[selected_job]
        option_cost = cost[states_dict[curr_state].jobs]
        l[curr_state] += sum(option_cost)

    V = np.dot(np.linalg.inv(np.eye(pi_len) - gamma * P), l)
    return V, P


def policyIteration(policy, gamma):
    nextV, P = calcValueFunction(policy, gamma)
    conToNextIter = True
    nextPolicy = policy
    first_state_value = []

    while conToNextIter:
        prevV, _ = calcValueFunction(nextPolicy.astype(int), gamma)
        nextV = np.zeros(len(nextV))
        nextPolicy = np.zeros(len(policy))
        for state_idx in range(len(policy)):
            state = states_dict[state_idx]
            possible_states = state.next_states
            possible_jobs = state.jobs
            if len(possible_states) == 0:
                nextV[state_idx] = 0
                nextPolicy[state_idx] = 1
            else:
                new_p = P[possible_states, :]
                new_v = prevV
                v_for_curr_state = state.loss + gamma * np.dot(new_p, new_v)
                nextV[state_idx] = np.min(v_for_curr_state)

                nextPolicy[state_idx] = possible_jobs[np.argmin(v_for_curr_state)] + 1

        first_state_value.append(nextV[31])
        conToNextIter = np.any(np.logical_not(np.equal(prevV, nextV)))

    return nextPolicy, first_state_value


if __name__ == '__main__':
    init(5)
    c_max_policy = buildCMaxPolicy('cost')
    V_c_max = calcValueFunction(c_max_policy, 0.9)

    plt.figure()
    plt.stem(range(1, 32), c_max_policy[1:])
    plt.xlabel('state'), plt.ylabel('action')
    plt.show()

    opt_policy, first_state = policyIteration(c_max_policy, 0.9)
    plt.figure()
    plt.stem(range(1, len(first_state) + 1), first_state)
    plt.show()

    mc_max_policy = buildCMaxPolicy('mc')

    plt.figure()
    plt.stem(range(1, 32), opt_policy[1:])
    plt.hold(True)
    points, lines, _ = plt.stem(range(1, 32), mc_max_policy[1:])
    plt.setp(lines, color='r')
    plt.setp(points, color='r')
    plt.show()

    print('all done')
