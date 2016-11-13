# coding:utf-8

import numpy as np


def exp_orderings(y_true, y_selected, oor_dist_fine=10.0):
    length = len(y_true)
    labels = set(y_true)

    error = 0.0
    normalizer = 0.0

    for index, y in enumerate(y_selected):
        if y in labels:
            true_idx = np.argwhere(np.equal(y_true, y)).flatten()[0]
            error += np.log2(max(index - true_idx + 1, 1)) * (length - index)
        else:
            error += np.log2(oor_dist_fine) * (length - index)
        normalizer += np.log2(oor_dist_fine) * (length - index)

    return 1.0 - error / normalizer


if __name__ == "__main__":
    print(exp_orderings([4, 790, 3, 36], [44, 790]))
