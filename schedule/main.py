import numpy as np

RAND_RANGE=1000
if __name__ == '__main__':
    a=np.random.randint(
        1, RAND_RANGE) / float(RAND_RANGE)
    print(a)