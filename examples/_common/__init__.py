"""Shared code of the example scripts: command line, results layout, run driver."""

import os

# Double precision everywhere; must be set before JAX is imported.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from .layout import METHOD_DIRS, results_tags  # noqa: E402,F401
from .methods import ALL_METHODS, Settings, canonical, make_method  # noqa: E402



def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--method", default="nystrom_gaussian", help=", ".join(ALL_METHODS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=None, help="override the budget")
    p.add_argument("--results-root", default="results")
    p.add_argument("--platform", default=os.environ.get("JAX_PLATFORMS", "gpu"))
    p.add_argument("--trace", default=None, help="write a step-by-step trace (JSON)")
    p.add_argument(
        "--trace-native-cadence",
        action="store_true",
        help="with --trace, keep the method's logging cadence (default: log every step)",
    )
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="FIELD=VALUE",
        help="override any field of Settings, e.g. --set gamma=100 ell_max=300",
    )
    return p


def setup_jax(platform: str):
    import jax

    jax.config.update("jax_enable_x64", True)
    if platform:
        jax.config.update("jax_platform_name", platform)


def apply_overrides(settings: Settings, overrides: list[str]) -> Settings:
    fields = {f.name for f in dataclasses.fields(Settings)}
    kw = {}
    for item in overrides:
        name, _, value = item.partition("=")
        if name not in fields:
            raise SystemExit(f"Unknown setting {name!r}")
        try:
            kw[name] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            kw[name] = value
    return settings.replace(**kw)


PACKAGES = ("randngd", "ssbroyden-optimistix", "jax", "jaxlib", "equinox", "optimistix",
            "lineax", "optax", "flax", "numpy", "scipy")


def environment() -> dict:
    """What a run depends on besides its settings: command line, package versions,
    git commit of this checkout, XLA flags and device."""
    from importlib import metadata

    import jax

    versions = {"python": sys.version.split()[0]}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=root, capture_output=True, text=True).stdout.strip())
        commit = sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        commit = None
    device = jax.devices()[0]
    return {
        "argv": sys.argv,
        "versions": versions,
        "commit": commit,
        "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
        "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
        "device": {"platform": device.platform, "kind": device.device_kind,
                   "count": jax.device_count()},
    }


def run(problem: SimpleNamespace, settings: Settings, args) -> None:
    """Train ``problem`` with ``args.method`` and log under ``args.results_root``.

    ``problem`` provides ``loss``, ``params``, ``gramians`` (metric functions by
    name), ``error_kwargs``, ``results_base``, ``seed_dir``, ``inputs`` and optionally
    ``time_dependent``, ``mean_zero_flags``, ``error_batch_size``.
    """
    from .train import TrainConfig, train_optax, train_optimistix
    from .trace import Tracer

    method = canonical(args.method)
    kind, optimizer, maxiter, showevery, logerrorevery = make_method(
        method, settings, problem.gramians
    )
    if args.maxiter is not None:
        maxiter = args.maxiter
    params_tag, suffixes = results_tags(settings, problem)
    for key, extra in getattr(problem, "method_dir_suffix", {}).items():
        suffixes[key] = suffixes.get(key, "") + extra
    log_dir = os.path.join(
        args.results_root,
        problem.results_base,
        params_tag,
        problem.seed_dir,
        METHOD_DIRS[method] + suffixes.get(method, ""),
    )
    config = TrainConfig(
        maxiter=maxiter,
        showevery=showevery,
        logerrorevery=logerrorevery,
        log_dir=log_dir,
        time_dependent=getattr(problem, "time_dependent", False),
        mean_zero_flags=getattr(problem, "mean_zero_flags", False),
        error_batch_size=getattr(problem, "error_batch_size", None),
    )
    tracer = None
    if args.trace:
        tracer = Tracer(args.trace, meta={"method": method, "seed": settings.seed})
        tracer.record_inputs(**problem.inputs)
        if not args.trace_native_cadence:
            config.logerrorevery = 1

    print(f"method={method} seed={settings.seed} maxiter={maxiter} -> {log_dir}")
    train = train_optimistix if kind == "optimistix" else train_optax
    result = train(
        optimizer, problem.loss, problem.params, config, **problem.error_kwargs,
        tracer=tracer,
    )
    with open(os.path.join(result.run_dir, "run_info.json"), "w") as f:
        json.dump(
            {
                "method": method,
                "settings": {
                    k: (v if isinstance(v, (int, float, str, bool, type(None), dict)) else repr(v))
                    for k, v in dataclasses.asdict(settings).items()
                },
                "maxiter": maxiter,
                "iterations": result.iterations,
                "final_loss": result.final_loss,
                "total_time": result.total_time,
                "stop_reason": result.stop_reason,
                "environment": environment(),
            },
            f,
            indent=1,
        )
