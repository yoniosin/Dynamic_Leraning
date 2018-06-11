from typing import Any, Union

import numpy as np
import matplotlib.pyplot as plt
from random import random

mu = np.asarray([0.6, 0.5, 0.3, 0.7, 0.1])
# mu = np.asarray([0.6, 0.5, 0.3])
cost = np.asarray([1, 4, 6, 2, 9])
# cost = np.asarray([1, 4, 6])
policy_len = 2 ** len(cost)
mc = mu * cost


class State:
    def __init__(self, idx):
        self.idx = idx
        self.next_states = []
        self.jobs = []
        self.const_loss = 0

    def calcLoss(self):
        self.const_loss = sum(cost[self.jobs])

    def calcValFunc(self, value_func, gamma):
        val_on_success = mu[self.jobs] * value_func[self.next_states]
        val_on_fail = (1 - mu[self.jobs]) * value_func[self.idx]
        return self.const_loss + gamma * (val_on_success + val_on_fail)


def Simulate(state_idx, action):
    state = states_dict[state_idx]
    if action not in state.jobs:
        raise ValueError

    thres = mu[action]
    if random(1) < thres:  # job completed w.p mu
        next_state = calcNextState(state.idx, action)
    else:
        next_state = state.idx

    return state.const_loss, next_state


states_dict = {i: State(i) for i in range(policy_len)}


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
    tmp = np.linalg.inv(np.eye(pi_len) - gamma * P)
    V = np.dot(tmp, l)
    V[0] = 0
    return V, P


def policyIteration(policy, gamma):
    nextV, P = calcValueFunction(policy, gamma)
    conToNextIter = True
    nextPolicy = policy
    first_state_value = []

    while conToNextIter:
        prevV = nextV
        evaluateV = np.zeros(len(prevV))
        nextPolicy = np.zeros(len(policy))
        for state_idx in range(len(policy)):
            state = states_dict[state_idx]
            possible_states = state.next_states
            possible_jobs = state.jobs
            if len(possible_states) == 0:
                evaluateV[state_idx] = 0
                nextPolicy[state_idx] = 1
            else:
                v_for_curr_state = state.calcValFunc(prevV, gamma)
                evaluateV[state_idx] = np.min(v_for_curr_state)

                nextPolicy[state_idx] = possible_jobs[np.argmin(v_for_curr_state)] + 1

        first_state_value.append(evaluateV[-1])
        nextV, _ = calcValueFunction(nextPolicy.astype(int), gamma)
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
    plt.stem(range(1, policy_len), opt_policy[1:], '-b', label='V')
    plt.hold(True)
    plt.stem(range(1, policy_len), mc_max_policy[1:], '-g', label='mc')
    plt.legend(loc='upper left')
    plt.show()

    print('all done')
