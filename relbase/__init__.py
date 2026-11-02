"""relbase: reliability-matched baselines for between-person overlap of normative deviations."""
from .core import (
    METRICS,
    BaselineModel,
    discriminability,
    estimate_inputs,
    extreme_indicator,
    icc21,
    identification_rate,
    implied_shared_fraction,
    marginal_quantiles,
    max_prevalence,
    mean_jaccard,
    mean_prevalence,
    spearman_brown,
)

__version__ = "0.1.2"
__all__ = ["METRICS", "BaselineModel", "discriminability", "estimate_inputs", "extreme_indicator", "icc21", "identification_rate", "implied_shared_fraction", "marginal_quantiles",
           "max_prevalence", "mean_jaccard", "mean_prevalence", "spearman_brown"]
