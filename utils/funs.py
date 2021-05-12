"""
Utility functions for classification/clustering
"""

import functools

def flatten_tensors(xs: list) -> list:
    """
    Flattening list of tensors to list.
    """
    return list(functools.reduce(lambda x, y: x + y, map(lambda l: l.tolist(), xs)))

def flatten(xs: list) -> list:
<<<<<<< Updated upstream
    """"
    Flattening list of lists to list.
    """
=======
>>>>>>> Stashed changes
    return list(functools.reduce(lambda x, y: x + y, xs))
