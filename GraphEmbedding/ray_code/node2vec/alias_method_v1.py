# https://www.bilibili.com/video/BV1jU4y1H7W5/?spm_id_from=333.337.search-card.all.click&vd_source=d8d2c837a6fad4a93cc95f349e30e2fe
import numpy as np
import numpy.random as npr


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=int)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    K = len(J)

    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand() * K))

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:
        return kk
    else:
        return J[kk]


K = 5
N = 1000

# Get a random probability vector.
probs = npr.dirichlet(np.ones(K), 1).ravel()

# Construct the table.
J, q = alias_setup(probs)

# Generate variates.
X = np.zeros(N)
for nn in range(N):
    X[nn] = alias_draw(J, q)

print(X)
