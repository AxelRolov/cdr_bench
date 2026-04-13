from __future__ import annotations

from importlib import import_module

_PARAM_EXPORTS = {
    "ScoringParams",
    "OptimizerParams",
    "DimReducerParams",
    "PCAParams",
    "UMAPParams",
    "TSNEParams",
    "GTMParams",
}

_OPT_EXPORTS = {
    "Optimizer",
    "perform_optimization",
    "create_param_grid",
}

__all__ = sorted(_PARAM_EXPORTS | _OPT_EXPORTS)


def __getattr__(name: str):
    if name in _PARAM_EXPORTS:
        module = import_module(".params", __name__)
        return getattr(module, name)
    if name in _OPT_EXPORTS:
        module = import_module(".optimization", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
