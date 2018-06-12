import numpy as np
import matplotlib.pyplot as plt
from random import random


class State:
    def __init__(self, idx):
        self.idx = idx
        self.next_states = []
        self.jobs = []
        self.const_loss = 0

    def isJobWaiting(self, job):
        return (self.idx & (1 << job)) >> job == 1

    def calcNextState(self, job):
        return 0 if self.idx == 0 else self.idx - 2 ** job

    def CalcPosJobs(self, job_num):
        for job in range(job_num):
            if self.isJobWaiting(job):
                next_state = self.calcNextState(job)
                self.next_states.append(next_state)
                self.jobs.append(job)

    def calcLoss(self, job_num, cost_vec):
        self.CalcPosJobs(job_num)
        self.const_loss = sum(cost_vec[self.jobs])

    def calcValFunc(self, value_func, gamma):
        success_vec = mu[self.jobs] * value_func[self.next_states]
        fail_vec = (1 - mu[self.jobs]) * value_func[self.idx]
        return self.const_loss + gamma * (success_vec + fail_vec)

    def Simulate(self, action, mu_vec):
        if action not in self.jobs:
            raise ValueError

        thres = mu_vec[action]
        next_state = self.calcNextState(action) if random() < thres else self.idx  # job completed w.p mu

        return self.const_loss, next_state


class Queue:
    def __init__(self, cost_vec, mu_vec):
        self.states_dict = {}
        self.cost = cost_vec
        self.mu = mu_vec
        self.mc = self.mu * self.cost
        self.job_num = len(cost_vec)
        self.policy_len = 2 ** self.job_num

        for state_idx in range(self.policy_len):
            self.states_dict[state_idx] = State(state_idx)
            state = self.states_dict[state_idx]
            state.calcLoss(self.job_num, self.cost)

    def buildCMaxPolicy(self, calc_type):
        policy = np.ones(self.policy_len)
        cost_vec = self.mc if calc_type == 'mc' else self.cost

        for state_idx in range(1, self.policy_len):
            curr_state = self.states_dict[state_idx]
            option_cost = cost_vec[curr_state.jobs]
            chosen_s = curr_state.jobs[np.argmax(option_cost)]
            policy[state_idx] = chosen_s + 1

        return policy.astype(int)

    def calcValueFunction(self, policy, gamma):
        policy = np.asarray(policy, int) - 1
        pi_len = len(policy)
        P = np.zeros((pi_len, pi_len))
        l = np.zeros(pi_len)

        for curr_state_idx, selected_job in enumerate(policy):
            curr_state = self.states_dict[curr_state_idx]
            P[curr_state_idx, curr_state_idx] = 1 - self.mu[selected_job]
            next_state = self.states_dict[curr_state.calcNextState(selected_job)]
            P[curr_state_idx, next_state.idx] += self.mu[selected_job]
            option_cost = self.cost[curr_state.jobs]
            l[curr_state_idx] += sum(option_cost)
        tmp = np.linalg.inv(np.eye(pi_len) - gamma * P)
        V = np.dot(tmp, l)
        V[0] = 0
        return V, P

    def policyIteration(self, policy, gamma):
        nextV, P = self.calcValueFunction(policy, gamma)
        nextPolicy = policy
        first_state_value = []

        for _ in range(10):
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
            nextV, _ = self.calcValueFunction(nextPolicy.astype(int), gamma)

        return nextPolicy, first_state_value

    def Simulate(self, state_idx, action):
        return self.states_dict[state_idx].Simulate(action, self.mu)


if __name__ == '__main__':
    mu = np.asarray([0.6, 0.5, 0.3, 0.7, 0.1])
    cost = np.asarray([1, 4, 6, 2, 9])

    queue = Queue(cost, mu)
    c_max_policy = queue.buildCMaxPolicy('cost')
    V_c_max, _ = queue.calcValueFunction(c_max_policy, 0.9)

    plt.figure()
    plt.stem(range(1, 32), c_max_policy[1:])
    plt.xlabel('state'), plt.ylabel('action')
    plt.title('policy of max cost of the remaining jobs')
    plt.show()

    opt_policy, first_state = queue.policyIteration(c_max_policy, 0.9)
    V_opt_policy, _ = queue.calcValueFunction(opt_policy, 0.9)
    plt.figure()
    plt.stem(range(1, len(first_state) + 1), first_state)
    plt.xlabel('iteration'), plt.ylabel('value function for first state')
    plt.title('evaluation of the value function for first state')
    plt.show()

    mc_max_policy = queue.buildCMaxPolicy('mc')
    V_mc_max = queue.calcValueFunction(mc_max_policy, 0.9)

    plt.figure()
    plt.stem(range(1, queue.policy_len), opt_policy[1:], '-b', label='V opt')
    plt.hold(True)
    plt.stem(range(1, queue.policy_len), mc_max_policy[1:], '-g', label='V mc')
    plt.legend(loc='upper right')
    plt.xlabel('state'), plt.ylabel('action')
    plt.title('theoretic and calculated optimal policy')
    plt.show()

    plt.figure()
    plt.stem(range(1, queue.policy_len), V_opt_policy[1:], '-b', label='V opt')
    plt.hold(True)
    plt.stem(range(1, queue.policy_len), V_c_max[1:], '-g', label='V mc')
    plt.legend(loc='upper right')
    plt.xlabel('state'), plt.ylabel('value function')
    plt.title('Value function of theoretic and calculated optimal policy')
    plt.show()

    print('all done')
