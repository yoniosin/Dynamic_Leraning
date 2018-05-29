import numpy as np


def T(v1, r1, r2, p1, p2, gamma):
    p1_vec = np.array([1-p1, p1])
    p2_vec = np.array([p2, 1-p2])
    elem1 = r1 + gamma*(np.dot(p1_vec, v1))
    elem2 = r2 + gamma*(np.dot(p2_vec, v1))
    return np.array([elem1, elem2])


if __name__ == '__main__':
    v1 = np.random.rand(2)
    v2 = np.random.rand(2)
    r1 = r2 = 1
    p1 = 0.9
    p2 = 0.1
    gamma = 0.9
    TV1 = T(v1, r1, r2, p1, p2, gamma)
    TV2 = T(v2, r1, r2, p1, p2, gamma)

    subT = TV1 - TV2
    subVec = v1 - v2
    print('TSub ' + str(np.linalg.norm(subT)) + '\n' + 'VecSub ' + str(np.linalg.norm(subVec)))
    print('all done :)')