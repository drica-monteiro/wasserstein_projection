import numpy as np


def compute_quantile_bounds(values, alpha=0.05):
    """
    Compute mean and quantile bounds used to define stress range.
    """

    m = np.mean(values)
    q_low, q_high = np.quantile(values, [alpha, 1 - alpha])

    return m, q_low, q_high


def tau_to_t(tau, m, q_low, q_high):
    """
    Map stress parameter tau ∈ [-1,1] to target statistic t(τ).
    """

    if tau < 0:
        return m + tau * (m - q_low)

    elif tau > 0:
        return m + tau * (q_high - m)

    else:
        return m


def compute_lambda(m, t):
    """
    Compute lagrange multiplier lambda.
    """

    return 2 * (t - m)


def compute_prediction_portions(preds, num_classes):
    """
    Compute portion of predictions for each class.
    """

    return np.array([
        (preds == c).mean()
        for c in range(num_classes)
    ])


def run_stress_experiment(
        taus,
        base_stat,
        q_low,
        q_high,
        transform_fn,
        predict_fn,
        num_classes
):

    results = []
    projected_stats = []

    for tau in taus:

        t_tau = tau_to_t(tau, base_stat, q_low, q_high)
        delta = t_tau - base_stat

        data_proj = transform_fn(delta)
        preds = predict_fn(data_proj)

        portions = [
            (preds == c).mean()
            for c in range(num_classes)
        ]

        results.append(portions)
        projected_stats.append(t_tau)

    return np.array(results), np.array(projected_stats)