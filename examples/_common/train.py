"""Training loops used by the examples (Optimistix solvers and Optax optimizers).

Measurement protocol, as in the paper's experiments:

- The jitted solver step is compiled with two warm-up steps whose results are discarded.
- Each iteration is timed as ``step + terminate`` until ``block_until_ready``; error
  evaluation and logging are not timed.
- Iterations count *accepted* line-search steps; the time of rejected trials is
  included.
- If the parameters become NaN, the previous parameters are restored and training
  stops.
- Errors relative to the norms of the exact solution are logged to TensorBoard.
"""

import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import jacrev, vmap
from tensorboardX import SummaryWriter


@dataclass
class TrainConfig:
    maxiter: int = 1000
    showevery: int = 10
    logerrorevery: int = 1
    log_dir: str = "runs/default"
    time_dependent: bool = False
    mean_zero_flags: bool | list[bool] = False
    error_batch_size: int | None = None


@dataclass
class TrainResult:
    params: Any
    final_loss: float
    total_time: float
    iterations: int
    stop_reason: str = "maxiter"
    run_dir: str = ""


# =============================================================================
# Errors
# =============================================================================


def _batched_reduce(f, x, batch_size):
    """Accumulate ``sum(f(x)**2)``, ``sum(f(x))`` and the count over batches."""
    n = x.shape[0]
    n_chunks = int(np.ceil(n / batch_size))
    v_f = vmap(f)
    sum_sq = jnp.zeros((), dtype=jnp.result_type(float))
    sum_val = jnp.zeros((), dtype=jnp.result_type(float))
    count = 0
    for i in range(n_chunks):
        vals = v_f(x[i * batch_size : (i + 1) * batch_size])
        sum_sq = sum_sq + jnp.sum(vals**2.0)
        sum_val = sum_val + jnp.sum(vals)
        count += vals.size
    return sum_sq, sum_val, count


def compute_errors(
    models,
    params,
    u_star_list,
    x_eval,
    time_dependent=False,
    mean_zero_flags=False,
    batch_size=None,
):
    """``(L2, H1)`` errors of each model, with ``H1 = L2 + RMS(|grad e|)``.

    With ``time_dependent``, the gradient is taken in space only (first coordinate is
    time). With ``mean_zero_flags``, the error's mean is subtracted (e.g. pressure).
    """
    if isinstance(mean_zero_flags, bool):
        mean_zero_flags = [mean_zero_flags] * len(models)

    errors = []
    for model, p, u_star, is_meanzero in zip(models, params, u_star_list, mean_zero_flags):
        if u_star is None:
            errors.append((jnp.nan, jnp.nan))
            continue

        error = lambda x: model(p, x) - u_star(x)

        if time_dependent:

            def abs_grad(txyz):
                t = txyz[0:1]
                xyz = txyz[1:]
                g = jacrev(lambda xi: error(jnp.concatenate([t, xi])))(xyz)
                return jnp.sum(g**2.0) ** 0.5

        else:

            def abs_grad(x):
                return jnp.sum(jacrev(error)(x) ** 2.0) ** 0.5

        if batch_size is None:
            v_error = vmap(error)

            if is_meanzero:
                c = jnp.mean(v_error(x_eval))
                error = lambda x: model(p, x) - u_star(x) - c
                v_error = vmap(error)

            v_error_abs_grad = vmap(abs_grad)

            def l2_norm(f):
                return jnp.mean((f(x_eval)) ** 2.0) ** 0.5

            l2_error = l2_norm(v_error)
            h1_error = l2_error + l2_norm(v_error_abs_grad)
        else:
            if is_meanzero:
                _, sum_val, count = _batched_reduce(error, x_eval, batch_size)
                c = sum_val / count
                error = lambda x: model(p, x) - u_star(x) - c

            sum_sq_err, _, count_err = _batched_reduce(error, x_eval, batch_size)
            sum_sq_grad, _, count_grad = _batched_reduce(abs_grad, x_eval, batch_size)
            l2_error = (sum_sq_err / count_err) ** 0.5
            h1_error = l2_error + (sum_sq_grad / count_grad) ** 0.5

        errors.append((l2_error, h1_error))

    return errors


# =============================================================================
# Logging
# =============================================================================


def log_metrics(writer, iteration, elapsed_time, tot_time, loss_val, rel_errors, names,
                extra_metrics=None, metric_names=("L2", "H1")):
    writer.add_scalar("time/tot_time", tot_time, iteration)
    writer.add_scalar("time/iteration", elapsed_time, iteration)
    writer.add_scalar("loss", loss_val, iteration)
    for (err0, err1), name in zip(rel_errors, names):
        writer.add_scalar(f"error/{name}_{metric_names[0]}", float(err0), iteration)
        writer.add_scalar(f"error/{name}_{metric_names[1]}", float(err1), iteration)
    if extra_metrics:
        for key, value in extra_metrics.items():
            writer.add_scalar(f"solver/{key}", float(value), iteration)


def print_metrics(iteration, loss_val, rel_errors, names, metric_names=("L2", "H1")):
    error_strs = [
        f"{name}: {metric_names[0]}={e0:.3e}, {metric_names[1]}={e1:.3e}"
        for name, (e0, e1) in zip(names, rel_errors)
    ]
    print(f"[{iteration:5d}] loss={loss_val:.4e} | " + " | ".join(error_strs))


def state_has_nan(state) -> bool:
    nan_tree = jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), state)
    return jax.tree_util.tree_reduce(lambda a, b: a | b, nan_tree, initializer=False)


class _Context:
    def __init__(self, loss_fn, init_params, config, models, u_star_list, x_eval,
                 names, custom_error_fns, metric_names, tracer):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = f"{config.log_dir}/{timestamp}"
        self.writer = SummaryWriter(self.run_dir)
        self.custom_error_fns = custom_error_fns
        use_custom = custom_error_fns is not None and len(custom_error_fns) > 0
        self.compute_error = use_custom or (
            models is not None and u_star_list is not None and x_eval is not None
        )
        if use_custom:
            names = list(custom_error_fns.keys())
        elif names is None:
            names = [f"model_{i}" for i in range(len(models))] if models else []
        self.names = names
        if isinstance(config.mean_zero_flags, bool):
            self.mean_zero_flags = (
                [config.mean_zero_flags] * len(models) if models else []
            )
        else:
            self.mean_zero_flags = config.mean_zero_flags
        self.norms = None
        if self.compute_error and not use_custom:
            zero_params = jax.tree_util.tree_map(jnp.zeros_like, init_params)
            self.norms = compute_errors(
                models,
                zero_params if isinstance(zero_params, tuple) else (zero_params,),
                u_star_list,
                x_eval,
                config.time_dependent,
                self.mean_zero_flags,
                batch_size=config.error_batch_size,
            )
        self.models, self.u_star_list, self.x_eval = models, u_star_list, x_eval
        self.time_dependent = config.time_dependent
        self.showevery, self.logerrorevery = config.showevery, config.logerrorevery
        self.error_batch_size = config.error_batch_size
        self.metric_names = metric_names
        self.tracer = tracer

    def compute_and_log(self, initial_iter, iteration, loss_fn, params, elapsed,
                        tot_time, extra_metrics=None):
        should_log = iteration % self.logerrorevery == 0 or initial_iter
        should_print = iteration % self.showevery == 0 or initial_iter
        if not (should_log or should_print):
            return False
        loss_val = loss_fn(params)
        if self.custom_error_fns is not None and len(self.custom_error_fns) > 0:
            rel_errors = [fn(params) for fn in self.custom_error_fns.values()]
        elif self.compute_error:
            errors = compute_errors(
                self.models,
                params if isinstance(params, tuple) else (params,),
                self.u_star_list,
                self.x_eval,
                self.time_dependent,
                self.mean_zero_flags,
                batch_size=self.error_batch_size,
            )
            rel_errors = [(e[0] / n[0], e[1] / n[1]) for e, n in zip(errors, self.norms)]
        else:
            rel_errors = []
        if should_log:
            if self.tracer is not None:
                self.tracer.on_log(iteration, loss_val, rel_errors, self.names, extra_metrics)
            log_metrics(self.writer, iteration, elapsed, tot_time, loss_val, rel_errors,
                        self.names, extra_metrics, metric_names=self.metric_names)
        if should_print:
            print_metrics(iteration, loss_val, rel_errors, self.names, self.metric_names)
        return True

    def finalize(self, loss_fn, params, start_time, iterations, stop_reason):
        end = time.time()
        final_loss = float(loss_fn(params))
        print("=" * 60)
        print("OPTIMIZATION COMPLETE")
        print(f"  Total time: {end - start_time:.2f}s")
        print(f"  Iterations: {iterations}")
        print(f"  Final loss: {final_loss:.4e}")
        print(f"  Stopped by: {stop_reason}")
        print("=" * 60)
        self.writer.close()
        if self.tracer is not None:
            self.tracer.finish()
        return TrainResult(
            params, final_loss, end - start_time, iterations, stop_reason, self.run_dir
        )


def _solver_metrics(state):
    """Metrics logged for the preconditioner, if the solver has one."""
    ps = getattr(state, "preconditioner_state", None)
    if ps is None:
        return {}
    diag = ps.diagnostics() if hasattr(ps, "diagnostics") else {}
    extra = {}
    for key in ("ell", "mu", "rho", "nu", "ell_rounded"):
        if key in diag:
            extra[f"Gramian/{key}"] = float(diag[key])
    return extra


# =============================================================================
# Loops
# =============================================================================


def train_optimistix(solver, loss_fn, init_params, config, models=None, u_star_list=None,
                     x_eval=None, names=None, custom_error_fns=None,
                     metric_names=("L2", "H1"), tracer=None):
    """Train with an Optimistix minimiser ``solver``, driving ``step`` manually."""
    ctx = _Context(loss_fn, init_params, config, models, u_star_list, x_eval, names,
                   custom_error_fns, metric_names, tracer)

    def loss_optimistix(params, args):
        return loss_fn(params), None

    args = None
    f_struct = jax.ShapeDtypeStruct((), jnp.float64)
    aux_struct = None
    tags = frozenset()
    options = {}

    step = eqx.Partial(solver.step, fn=loss_optimistix, options=options, tags=tags)
    step = eqx.filter_jit(step)
    terminate = eqx.Partial(solver.terminate, fn=loss_optimistix, options=options,
                            tags=tags)
    terminate = eqx.filter_jit(terminate)

    params = deepcopy(init_params)
    state = solver.init(loss_optimistix, params, args, options, f_struct, aux_struct, tags)
    done, result = terminate(y=params, state=state, args=args)

    print("Compiling solver...")
    p_dummy, state_dummy, _ = step(y=params, state=state, args=args)
    _ = terminate(y=p_dummy, state=state_dummy, args=args)
    p_dummy, state_dummy, _ = step(y=p_dummy, state=state_dummy, args=args)
    _ = terminate(y=p_dummy, state=state_dummy, args=args)
    print("Compilation done.")
    print("=" * 60)
    print(f"OPTIMIZATION STARTS (solver: {type(solver).__name__})")
    print("=" * 60)

    start = time.time()
    tot_time = 0.0
    iter_count = 0
    elapsed_since_last_logged = 0.0
    initial_iter = True
    step_size = 1.0
    prev_stepsize = 1.0
    stop_reason = "maxiter"
    aux = None

    while not done:
        old_params = params
        iter_start = time.time()

        params, state, aux = step(y=params, state=state, args=args)
        done, result = terminate(y=params, state=state, args=args)

        done.block_until_ready()
        elapsed = time.time() - iter_start
        tot_time += elapsed
        elapsed_since_last_logged += elapsed

        if tracer is not None:
            tracer.on_params(params)
        if state_has_nan(params):
            print("NaN detected! Reverting to previous params.")
            params = old_params
            stop_reason = "nan"
            break

        # Accepted step? (searches record this in their state)
        if hasattr(state, "search_state") and hasattr(state.search_state, "accepted"):
            accepted = state.search_state.accepted
        else:
            accepted = True

        # Step size of the trial just evaluated (only searches with ``step_size``)
        if hasattr(state, "search_state"):
            prev_stepsize = step_size
            ss = state.search_state
            if hasattr(ss, "step_size"):
                step_size = ss.step_size

        if accepted or initial_iter:
            iter_count += 1
            extra = {}
            if hasattr(state, "search_state"):
                extra["stepsize"] = prev_stepsize
            extra.update(_solver_metrics(state))
            logged = ctx.compute_and_log(initial_iter, iter_count, loss_fn, params,
                                         elapsed_since_last_logged, tot_time, extra)
            if logged:
                elapsed_since_last_logged = 0.0
            initial_iter = False

        if iter_count >= config.maxiter:
            done = True
        elif done:
            stop_reason = "solver"

    params, _, _ = solver.postprocess(loss_optimistix, params, aux, args, options, state,
                                      tags, result)
    return ctx.finalize(loss_fn, params, start, iter_count, stop_reason)


def train_optax(optimizer, loss_fn, init_params, config, models=None, u_star_list=None,
                x_eval=None, names=None, custom_error_fns=None,
                metric_names=("L2", "H1"), tracer=None):
    """Train with an Optax optimizer (e.g. Adam)."""
    ctx = _Context(loss_fn, init_params, config, models, u_star_list, x_eval, names,
                   custom_error_fns, metric_names, tracer)
    import optax

    state = optimizer.init(init_params)

    @jax.jit
    def step_standard(params, state):
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss_val

    print("=" * 60)
    print("OPTIMIZATION STARTS (optax)")
    print("=" * 60)

    params = deepcopy(init_params)
    start = time.time()
    tot_time = 0.0
    iteration = 0
    stop_reason = "maxiter"
    for iteration in range(1, config.maxiter + 1):
        iter_start = time.time()
        old_params = params
        params, state, _ = step_standard(params, state)
        params = jax.tree_util.tree_map(lambda x: x.block_until_ready(), params)
        elapsed = time.time() - iter_start
        tot_time += elapsed

        if tracer is not None:
            tracer.on_params(params)
        if state_has_nan(params):
            print("NaN detected! Reverting to previous params.")
            params = old_params
            stop_reason = "nan"
            break

        ctx.compute_and_log(iteration == 1, iteration, loss_fn, params, elapsed, tot_time)

    return ctx.finalize(loss_fn, params, start, iteration, stop_reason)
