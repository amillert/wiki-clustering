import functools


def reduce(xs: list) -> list:
    return list(functools.reduce(lambda x, y: x + y, map(lambda l: l.tolist(), xs)))
