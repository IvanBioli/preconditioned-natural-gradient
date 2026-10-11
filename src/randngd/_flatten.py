"""Flattening helpers for parameter pytrees."""

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.flatten_util import ravel_pytree


def flatten_pytrees_d(*pytrees):
    """Ravel and concatenate pytrees.

    Returns the flat vector, a jitted function mapping a flat vector back to the tuple
    of pytrees, and the split points.
    """
    flat_arrays = []
    unravel_fns = []
    lengths = []

    for pytree in pytrees:
        flat, unravel = ravel_pytree(pytree)
        flat_arrays.append(flat)
        unravel_fns.append(unravel)
        lengths.append(len(flat))
    unravel_fns = tuple(unravel_fns)
    flat_concat = jnp.concatenate(flat_arrays)
    split_points = np.cumsum(np.array(lengths[:-1])).tolist()

    @jit
    def retrieve_pytrees(flat):
        splits = jnp.split(flat, split_points)
        return tuple(unravel(split) for unravel, split in zip(unravel_fns, splits))

    return flat_concat, retrieve_pytrees, split_points


def flat(unravel, argnum=0):
    """Decorator: unravel argument ``argnum`` before calling the function."""

    def flatten(func):
        def flattened(*args, **kwargs):
            args = [arg for arg in args]
            args[argnum] = unravel(args[argnum])
            return func(*args, **kwargs)

        return flattened

    return flatten


def ensure_tuple(x):
    """Convert a scalar argnum to a tuple."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x,)


def flatten_pytrees_for_argnums(params_sample, argnums):
    """Return the unflattening function for the parameters at ``argnums``."""
    argnums = ensure_tuple(argnums)
    pytrees_to_flatten = [params_sample[argnum] for argnum in argnums]
    _, retrieve_pytrees, _ = flatten_pytrees_d(*pytrees_to_flatten)
    return retrieve_pytrees


def flat_multi(retrieve_pytrees, argnums):
    """Decorator: unflatten several arguments, passed flat at ``argnums[0]``."""
    argnums = ensure_tuple(argnums)

    def flatten(func):
        def flattened(*args, **kwargs):
            args = list(args)
            unraveled = retrieve_pytrees(args[argnums[0]])
            for i, argnum in enumerate(argnums):
                args[argnum] = unraveled[i]
            return func(*args, **kwargs)

        return flattened

    return flatten
