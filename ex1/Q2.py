import numpy as np
from pydash import at

# CONST.
let_to_idx = {'b': 0, 'B': 0, 'k': 1, 'K': 1, 'o': 2, 'O': 0, '-': 3}
idx_to_let = {0: 'b', 1: 'k', 2: 'o', 3: '-'}

prob_tbl = np.asarray([[0.1, 0.325, 0.25, 0.325], [0.4, 0, 0.4, 0.2], [0.2, 0.2, 0.2, 0.4], [1, 0, 0, 0]])


def most_probable_word(org_k):
    if org_k == 0:
        raise IOError('k == 0 is not valid')

    if org_k == 1:
        res = 'b'
        return {'str': res, 'prob': 1}

    return back_most_probable_word(org_k, org_k, {})


def back_most_probable_word(k, org_k, curr_state):
    # calculate the first step, possible to pass just from 'b' to any letter
    if k == 1:
        ps_op_prob = []
        for ns_idx in range(len(curr_state['prob'])):
            ps_op_prob.append(prob_tbl[let_to_idx['b'], ns_idx] * curr_state['prob'][ns_idx])
        max_idx = np.argmax(ps_op_prob)
        calc_str = 'b' + curr_state['str'][max_idx]
        calc_prob = ps_op_prob[max_idx]

        final_ans = {'str': calc_str, 'prob': calc_prob}
        return final_ans

    # calculate the end step, possible to pass from any letter to end of the word ('-')
    if k == org_k:
        curr_prob = prob_tbl[:-1, let_to_idx['-']]

        curr_str = at(idx_to_let, 0, 1, 2)  # TODO
        curr_state = {'str': curr_str, 'prob': curr_prob}

        return back_most_probable_word(k - 1, org_k, curr_state)

    # regular flow: not the first/last letter
    calc_str = []
    calc_prob = []

    for ps_idx in range(len(prob_tbl) - 1):
        ps_op_prob = []
        for ns_idx in range(len(curr_state['prob'])):
            ps_op_prob.append(prob_tbl[ps_idx, ns_idx] * curr_state['prob'][ns_idx])
        max_idx = np.argmax(ps_op_prob)
        calc_str.append(idx_to_let[ps_idx] + curr_state['str'][max_idx])
        calc_prob.append(ps_op_prob[max_idx])

    curr_state = {'str': calc_str, 'prob': calc_prob}

    return back_most_probable_word(k - 1, org_k, curr_state)


if __name__ == '__main__':
    print(most_probable_word(5))
