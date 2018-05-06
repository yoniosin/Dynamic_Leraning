import numpy as np


def minThrows(floors):
    aux_mat = {(1, 1): 1, (1, 2): 1, (0, 1): 0, (0,2): 0}
    for floor in range(1, floors + 1):
        aux_mat[(floor, 1)] = floor

    for k in range(2, floors + 1):
        throw_vec = []
        for floor in range(2, k + 1):
            tmp = max(aux_mat[(floor - 1, 1)], aux_mat[(k - floor, 2)])
            throw_vec.append(tmp)

        aux_mat[(floor, 2)] = 1 + min(throw_vec)
    return aux_mat[(floors, 2)]


if __name__ == '__main__':
    print(minThrows(4))
