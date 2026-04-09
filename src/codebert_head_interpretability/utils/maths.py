import math


def compute_entropy(dist):
    entropy = 0.0
    for p in dist.values():
        if p > 0:
            entropy -= p * math.log(p)
    return entropy
