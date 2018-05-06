import numpy as np


# Function Name: calcpairs
# Description: calculates the number of possible arrangements of N natural numbers which sums up to X
# Inputs: int X - required sum, int N- amount of natural integers (must also be a positive)
# Output: number of possible arrangements (0 if impossible)
def calcpairs(n, x):
    if n % 1 != 0 or x % 1 != 0 or n < 1:
        raise IOError("Expecting natural variables")
    elif x < n:
        return 0
    else:
        vec = np.ones(x)  # calcpairs(1,x) always equals to 1
        for i in range(1, n):  # first iteration was done in previous line
            vec = np.cumsum(vec[:-1])  # sum excludes calcpairs(1,x)
        return vec[-1]


if __name__ == '__main__':
    print(calcpairs(12, 800))
