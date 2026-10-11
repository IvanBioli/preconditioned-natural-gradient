# Domain samplers and differential operators taken from unified-pinns
# (https://github.com/MariusZeinhofer/unified-pinns, MIT License, copyright (c) 2024
# Marius Zeinhofer; see LICENSE-unified-pinns in this directory), as found in the fork
# https://github.com/IvanBioli/unified-pinns at commit 2cf79de. Only the parts used by
# the examples are kept; the code is unchanged, because the random collocation points
# (and hence all results) depend on it bit for bit.

from typing import Any

import jax
import jax.numpy as jnp
from jax import hessian, jacrev, random

class Hyperrectangle:
    """
    A product of intervals in R^d.

    The hyperrectangle is specified as a product of intervals.
    For example

    intervals = ((0., 1.), (0., 1.), (0., 1.))

    is the unit cube in R^3. The assumption is that intervals
    is convertable to an array of shape (d, 2).

    Note that no method for deterministic integration points is
    provided in this class. The Hyperrectangle is potentially a high
    dimensional object. Deterministic integration points should be
    implemented in child classes.

    Parameters
    ----------
    intervals
        An iterable of intervals, see example above.

    """

    def __init__(self, intervals):
        self._intervals = jnp.array(intervals)

        l_bounds = None
        r_bounds = None

        if jnp.shape(self._intervals) == (2,):
            l_bounds = self._intervals[0]
            r_bounds = self._intervals[1]

        else:
            l_bounds = self._intervals[:, 0]
            r_bounds = self._intervals[:, 1]

        self._l_bounds = jnp.reshape(
            jnp.asarray(l_bounds, dtype=float),
            shape=(-1),
        )

        self._r_bounds = jnp.reshape(
            jnp.asarray(r_bounds, dtype=float),
            shape=(-1),
        )

        if len(self._l_bounds) != len(self._r_bounds):
            raise ValueError(
                "[In constructor of Hyperrectangle]: intervals "
                "is not convertable to an array of shape (d, 2)."
            )

        if not jnp.all(self._l_bounds < self._r_bounds):
            raise ValueError(
                "[In constructor of Hyperrectangle]: The "
                "lower bounds must be smaller than the upper bounds."
            )

        self._dimension = len(self._l_bounds)

    def measure(self) -> float:
        return jnp.prod(self._r_bounds - self._l_bounds)

    def random_integration_points(self, key: Any, N: int = 50):
        """
        N uniformly drawn collocation points in the hyperrectangle.

        Parameters
        ----------
        key
            A random key from jax.random.PRNGKey(<int>).
        N=50: int
            Number of random points.

        """
        return random.uniform(
            key,
            shape=(N, self._dimension),
            minval=jnp.broadcast_to(
                self._l_bounds,
                shape=(N, self._dimension),
            ),
            maxval=jnp.broadcast_to(
                self._r_bounds,
                shape=(N, self._dimension),
            ),
        )

    def distance_function(
        self,
        x,
    ):
        """
        A smooth approximation of the distance fct to the boundary.

        Note that when using this function in implementations for
        loss functions one should explicitly vectorize it using
        for instance vmap(distance_function, (0)) to let it act on
        arrays of shape (n, d) and return (n,).

        Parameters
        ----------
        x: Float[Array, "d"]
            A single spatial point x of shape (d,) where d is the
            dimension of the Hyperrectangle.
        """

        return jnp.prod((x - self._l_bounds) * (x - self._r_bounds))


class HyperrectangleBoundary:
    """
    Boundary of a Hyperrectangle

    """

    def __init__(self, intervals):
        self._intervals = jnp.array(intervals)
        # add check
        self._hyperrectangle = Hyperrectangle(intervals)

    def measure(self):
        return jnp.prod(self._intervals[:, 1] - self._intervals[:, 0])

    def random_integration_points(self, key, N=50):
        x = self._hyperrectangle.random_integration_points(key, N)
        length = jnp.sum(self._intervals[:, 1] - self._intervals[:, 0])

        for i in range(0, N):
            # advance random key
            key_0, key_1 = jax.random.split(key, num=2)
            key = key_0

            # location is float in [0, sum of interval lengths]
            location = random.uniform(key_1, (), minval=0.0, maxval=length)
            rand_dim = in_interval(location, self._intervals)

            # 0 or 1 depending on side
            index = jax.random.randint(key_0, shape=(), minval=0, maxval=2)
            rand_side = self._intervals[rand_dim, index]

            # project to random sides of the Hypercubes boundary
            x = x.at[i, rand_dim].set(rand_side)

        return x

    def distance_function(self, x):
        pass


def in_interval(xi: float, intervals) -> int:
    """
    xi should be in [0,length] and the function returns the interval
    number xi lies in

    """
    xi = float(xi)
    start = 0.0
    end = 0.0
    for i, len_i in enumerate(intervals[:, 1] - intervals[:, 0]):
        end += float(len_i)
        if xi >= start and xi <= end:
            return i

    return ValueError(
        "[in_interval] Invalid value to determine " "which boundary to project to."
    )


class HyperrectangleParabolicBoundary:
    """
    Boundary of a Hyperrectangle

    """

    def __init__(self, intervals):
        self._intervals = jnp.array(intervals)
        # add check
        self._hyperrectangle = Hyperrectangle(intervals)

    def measure(self):
        return jnp.prod(self._intervals[:, 1] - self._intervals[:, 0])

    def random_integration_points(self, key, N=50):
        x = self._hyperrectangle.random_integration_points(key, N)
        length = jnp.sum(self._intervals[:, 1] - self._intervals[:, 0])

        for i in range(0, N):
            # advance random key
            key_0, key_1 = jax.random.split(key, num=2)
            key = key_0

            # location is float in [0, sum of interval lengths]
            location = random.uniform(key_1, (), minval=0.0, maxval=length)
            rand_dim = in_interval(location, self._intervals)

            # 0 or 1 depending on side
            index = jax.random.randint(key_0, shape=(), minval=0, maxval=2)
            rand_side = self._intervals[rand_dim, index]

            # project to random sides of the Hypercubes boundary
            x = x.at[i, rand_dim].set(rand_side)

        # take out points corresponding to final time
        check_final = []
        for i in range(len(x)):
            if x[i, 0] == 1.0:
                check_final.append(False)
            else:
                check_final.append(True)

        return x[check_final, :]

    def distance_function(self, x):
        pass

def laplace(func, argnum=0):
    """
    Computes laplacian of func with respect to the argument argnum.

    Parameters
    ----------
    func: Callable
        Function whose laplacian should be computed.

    argnum: int
        Argument number wrt which laplacian should be computed.

    Returns
    -------
    Callable of same signature as func.

    Issues
    ------
    Vector valued func. So far not tested if this function works
    appropriately for vector valued functions. We need an
    implementation that does this.

    """
    hesse = hessian(func, argnum)
    return lambda *args, **kwargs: jnp.trace(
        hesse(*args, **kwargs),
        axis1=-2,
        axis2=-1,
    )


def del_i(
    func,
    argnum: int = 0,
):
    """
    Partial derivative for a function of signature (d,) ---> ().

    """

    def func_splitvar(*args):
        return func(jnp.array(args))

    d_splitvar_di = jacrev(func_splitvar, argnum)

    def dfunc_di(x):
        return d_splitvar_di(*x)

    return dfunc_di


def div(func, argnum=0):
    def div_f(*args, **kwargs):
        J = jacrev(func, argnum)(*args, **kwargs)
        return jnp.trace(J, axis1=-2, axis2=-1)

    return div_f


