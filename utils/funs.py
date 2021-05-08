import functools


# TODO(amillert): Can they be generalized?
def reduce_tensors(xs: list) -> list:
    return list(functools.reduce(lambda x, y: x + y, map(lambda l: l.tolist(), xs)))

def reduce(xs: list) -> list:
    return list(functools.reduce(lambda x, y: x + y, xs))
