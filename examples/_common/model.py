"""The fully connected network used in the examples (Flax NNX).

The initialisation (Glorot-uniform kernels, biases uniform in [0, sqrt(3/width))) and
the parameter ordering of ``nnx.split`` fix the random pivots and sketches of the
preconditioners, so this module is kept exactly as in the experiments of the paper.
"""

from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers


class MLP_jax(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        rngs: nnx.Rngs,
        activation_layer: Callable | None = nnx.tanh,
        kernel_init: Callable | None = initializers.glorot_uniform(dtype=jnp.float64),
    ):
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            bias_init = initializers.uniform(
                jnp.sqrt(3 / hidden_dim), dtype=jnp.float64
            )
            layers.append(
                nnx.Linear(
                    in_dim,
                    hidden_dim,
                    rngs=rngs,
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    param_dtype=jnp.float64,
                )
            )
            layers.append(activation_layer)
            in_dim = hidden_dim
        bias_init = initializers.uniform(jnp.sqrt(3 / hidden_dim), dtype=jnp.float64)
        layers.append(
            nnx.Linear(
                in_dim,
                hidden_channels[-1],
                rngs=rngs,
                kernel_init=kernel_init,
                bias_init=bias_init,
                param_dtype=jnp.float64,
            )
        )

        self.layers = nnx.List(layers)

    def __call__(self, x):
        out = x
        for l in self.layers:
            out = l(out)
        return out


def create_model(layer_sizes: list[int], seed: int, activation_layer=nnx.tanh):
    """MLP ``[input, hidden..., output]``; returns ``(model, params)``.

    ``model(params, x)`` evaluates the network (output shape ``(layer_sizes[-1],)``);
    ``params`` is the ``nnx.State`` of its parameters.
    """
    rngs = nnx.Rngs(seed)
    uNN = MLP_jax(
        in_channels=layer_sizes[0],
        hidden_channels=layer_sizes[1:],
        activation_layer=activation_layer,
        rngs=rngs,
    )
    graph, params, _ = nnx.split(uNN, nnx.Param, ...)
    model = lambda p, x: nnx.call((graph, p))(x)[0]
    return model, params
