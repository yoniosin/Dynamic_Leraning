import numpy as np

helper_dict = {}
v_dict = {}
prob_vec = np.ones((1, 10), float)
prob_vec[0, 8] = 4
prob_vec /= 13

x_prob, x_val = prob_2cards()


def calcTerminalYProb(x, y):
    if (x, y) in helper_dict.keys():
        return helper_dict[(x, y)]

    if y > 21 or 17 <= y < x:
        helper_dict[(x, y)] = 1
        return 1

    if y > max(16, x):
        helper_dict[(x, y)] = -1
        return -1

    if y >= 17 and y == x:
        helper_dict[(x, y)] = 0
        return 0

    rewardVec = np.zeros(10)
    for i in range(10):
        rewardVec[i] = calcTerminalYProb(x, y + (i + 2))
    helper_dict[(x, y)] = sum(rewardVec * prob_vec)
    return helper_dict[(x, y)]


def ValIterationMDP(x,y):
    if (x, y) in v_dict.keys():
        return v_dict[(x, y)]

    if x >= 21:
        return calcTerminalYProb(x,y)

    # TODO: need to caclculate probabilities of the first sum of 2 cards
    # sum of first 2 cards of the player
    for x in x_val:
        # first card of the dealer
        for y in range(10):
            # every possible card the player get in this iteration
            hits_vec = np.zeros(10)
            for i in range(10):
                hits_vec[i] = ValIterationMDP(x + i, y)

            hits = sum(hits_vec * prob_vec)
            sticks = calcTerminalYProb(x,y)
            v_dict[(x, y)] = {'val' : max(hits, sticks), 'a' : 'hits' if hits > sticks else 'sticks'}


def prob_2cards():
    max_card = 11
    min_card = 2
    sum_prob = [0] * (2*(max_card - min_card)+1)
    sum_val = [0] * (2*(max_card - min_card)+1)

    for card1 in range(2,12):
        for card2 in range(2,12):
            prob1 = 4/13 if card1 == 10 else 1/13
            prob2 = 4/13 if card2 == 10 else 1/13
            sum_idx = card1 + card2 - 2*min_card
            sum_prob[sum_idx] += prob1*prob2
            sum_val[sum_idx] = card1 + card2
    return sum_prob, sum_val

if __name__ == '__main__':
#    print(calcTerminalYProb(2, 14))
    print(prob_2cards())
