import numpy as np
import matplotlib.pyplot as plt
from random import random

class State:
    def __init__(self, idx):
        self.idx = idx
        self.next_states = []
        self.jobs = []
        self.const_loss = 0

    def calcLoss(self):
        self.const_loss = sum(cost[self.jobs])

    def calcValFunc(self, value_func, gamma):
        success_vec = mu[self.jobs] * value_func[self.next_states]
        fail_vec = (1 - mu[self.jobs]) * value_func[self.idx]
        return self.const_loss + gamma * (success_vec + fail_vec)

    def isJobWaiting(self, job):
        if (self.idx & (1 << job)) >> job == 1:
            return True
        return False
    
    def calcNextState(self, job):
        if self.idx == 0:
            return 0
        return self.idx - 2 ** job
    
    def Simulate(self, action):
    if action not in self.jobs:
        raise ValueError

    thres = self.mu[action]
    if random(1) < thres:  # job completed w.p mu
        next_state = self.calcNextState(action)
    else:
        next_state = self.idx

    return state.const_loss, next_state



class Queue:
    def __init(self, cost, mu):
        self.states_dict = {}
        self.cost = cost
        self.mu = mu
        self.mc = self.mu * self.cost
        self.job_num = len(cost)
        self.policy_len = 2 ** self.job_num
        
        for state_idx in range(self.policy_len):
            self.states_dict[state_idx] = State(state_idx)
            state = self.states_dict[state_idx]
            for job in range(job_num):
                if isJobWaiting(state_idx, job):
                    next_state = calcNextState(state_idx, job)
                    state.next_states.append(next_state)
                    state.jobs.append(job)

           state.calcLoss()
        
    def buildCMaxPolicy(self, calc_type):
    policy = np.ones(self.policy_len)
    if calc_type == 'mc':
        cost_vec = self.mc
    else:
        cost_vec = self.cost
    for state in range(1, self.policy_len):
        curr_state = self.states_dict[state]
        option_cost = cost_vec[curr_state.jobs]
        chosen_s = curr_state.jobs[np.argmax(option_cost)]
        policy[state] = chosen_s + 1

    return policy.astype(int)


    def calcValueFunction(self, policy, gamma):
        policy = policy - 1
        pi_len = len(policy)
        P = np.zeros((pi_len, pi_len))
        l = np.zeros(pi_len)

        for curr_state_idx, selected_job in enumerate(policy):
            curr_state = self.states_dict[curr_state_idx]
            P[curr_state_idx, curr_state_idx] = 1 - mu[selected_job]
            next_state = curr_state.calcNextState(selected_job)
            P[curr_state_idx, next_state.idx] += mu[selected_job]
            option_cost = cost[curr_state.jobs]
            l[curr_state_idx] += sum(option_cost)
        tmp = np.linalg.inv(np.eye(pi_len) - gamma * P)
        V = np.dot(tmp, l)
        V[0] = 0
        return V, P


    def policyIteration(self, policy, gamma):
        nextV, P = self.calcValueFunction(policy, gamma)
        conToNextIter = True
        nextPolicy = policy
        first_state_value = []

        while conToNextIter:
            prevV = nextV
            evaluateV = np.zeros(len(prevV))
            nextPolicy = np.zeros(len(policy))
            for state_idx in range(len(policy)):
                state = self.states_dict[state_idx]
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
    mu = np.asarray([0.6, 0.5, 0.3, 0.7, 0.1])
    cost = np.asarray([1, 4, 6, 2, 9])

    queue = Queue(cost, mu)
    c_max_policy = queue.buildCMaxPolicy('cost')
    V_c_max = queue.calcValueFunction(c_max_policy, 0.9)

    plt.figure()
    plt.stem(range(1, 32), c_max_policy[1:])
    plt.xlabel('state'), plt.ylabel('action')
    plt.show()

    opt_policy, first_state = queue.policyIteration(c_max_policy, 0.9)
    plt.figure()
    plt.stem(range(1, len(first_state) + 1), first_state)
    plt.show()

    mc_max_policy = queue.buildCMaxPolicy('mc')

    plt.figure()
    plt.stem(range(1, policy_len), opt_policy[1:], '-b', label='V')
    plt.hold(True)
    plt.stem(range(1, policy_len), mc_max_policy[1:], '-g', label='mc')
    plt.legend(loc='upper left')
    plt.show()

    print('all done')
