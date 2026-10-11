"""Results layout of the original scripts (no JAX import: used by the runners too)."""

# Result folder names of each method (kept from the original experiments, so the
# analysis scripts read old and new runs alike).
METHOD_DIRS = {
    "ngd_full": "NGDFullPreconditioner",
    "ngd_matfree": "NGDMatfreePreconditioner",
    "rpchol": "NGDPCholPreconditioner",
    "nystrom_gaussian": "NGDNystromGaussianPreconditioner_svd",
    "nystrom_sampling": "NGDNystromSamplingPreconditioner_sampling_svd",
    "bfgs": "BFGS",
    "ssbroyden": "SSBroyden",
    "lbfgs": "LBFGS",
    "adam": "Adam",
}


def results_tags(settings, problem):
    """Directory tags of the original scripts for non-default settings.

    Returns ``(params_tag, method_suffixes)``: ``params_tag`` (e.g. ``"_factor100"``) is
    a directory level between the problem and the seed; the suffixes are appended to
    the method directories (RPChol: ``_atol..``, Nyström: ``_ratiofactor..``).
    ``problem.tag_factor`` (default True) and ``problem.tag_sampling_ratio`` (default
    False) follow the per-script differences.
    """
    tag = ""
    if getattr(problem, "tag_factor", True) and settings.gamma != 10.0:
        tag += f"_factor{settings.gamma:g}"
    if settings.atol_factor is not None:
        tag += f"_atolpchol{settings.atol_factor:g}"
    if settings.ratiofactor_ell != 10.0:
        tag += f"_ratioell{settings.ratiofactor_ell:g}"
    ratio = "" if settings.ratiofactor_ell == 10.0 else f"_ratiofactor{settings.ratiofactor_ell:g}"
    suffixes = {
        "rpchol": "" if settings.atol_factor is None else f"_atol{settings.atol_factor:g}",
        "nystrom_gaussian": ratio,
    }
    if getattr(problem, "tag_sampling_ratio", False):
        suffixes["nystrom_sampling"] = ratio
    return tag, suffixes
