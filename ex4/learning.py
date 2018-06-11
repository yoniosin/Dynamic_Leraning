import ex4.Planning as Qu
import numpy as np


if __name__ == '__main__':
    mu = np.asarray([0.6, 0.5, 0.3, 0.7, 0.1])
    cost = np.asarray([1, 4, 6, 2, 9])

    queue = Qu.Queue(cost, mu)
    loss, next_sate = queue.Simulate(31, 0)
    print('All Done :)')

