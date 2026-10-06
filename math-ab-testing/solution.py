import math

import numpy as np


def conversion_rate(conversions: int, visitors: int) -> float:
    """
    The observed conversion rate: fraction of visitors who converted.
    """
    return conversions / visitors


def two_proportion_z_test(
    conversions_a: int, visitors_a: int, conversions_b: int, visitors_b: int
) -> tuple[float, float]:
    """
    A/B testing's standard statistical test: are variant A's and
    variant B's conversion RATES (proportions, not continuous
    measurements like 03-hypothesis-testing's t-test) different
    enough to be more than noise?

    Uses a POOLED proportion (combining both groups) for the standard
    error, the correct approach under the null hypothesis that both
    groups truly share the same underlying conversion rate:

        p_pooled = (conversions_a + conversions_b) / (visitors_a + visitors_b)
        SE = sqrt(p_pooled * (1 - p_pooled) * (1/visitors_a + 1/visitors_b))
        z = (rate_a - rate_b) / SE
        p_value = 2 * (1 - normal_cdf(|z|))    (two-sided)

    `conversion_rate` is already provided above.
    """
    p_a = conversion_rate(conversions_a, visitors_a)
    p_b = conversion_rate(conversions_b, visitors_b)
    p_pooled = (conversions_a + conversions_b) / (visitors_a + visitors_b)

    standard_error = np.sqrt(p_pooled * (1.0 - p_pooled) * (1.0 / visitors_a + 1.0 / visitors_b))
    z_statistic = (p_a - p_b) / standard_error
    standard_normal_cdf = 0.5 * (1.0 + math.erf(abs(z_statistic) / math.sqrt(2.0)))
    p_value = 2.0 * (1.0 - standard_normal_cdf)
    return float(z_statistic), float(p_value)
