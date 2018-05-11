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


def findOnesSubMat(binMat):

    matSize = binMat.shape
    onesMat = np.zeros(matSize, dtype=np.uint)

    onesMat[:, 0] = binMat[:, 0]
    onesMat[0, :] = binMat[0, :]

    for i in range(1, matSize[0]):
        for j in range(1, matSize[1]) :
            minNeighbVal = min(onesMat[i - 1, j], onesMat[i, j - 1], onesMat[i - 1, j - 1])
            onesMat[i, j] = binMat[i, j] * (1 + minNeighbVal)
    print('final Mat')
    print(onesMat)
    return onesMat.max()


if __name__ == '__main__':
    # print(minThrows(4))
    N = 10
    p = 0.9
    binMat = np.random.choice(a=[1, 0], size=(N, N), p=[p, 1 - p])
    print('original Mat')
    print(binMat)
    print(findOnesSubMat(binMat))

