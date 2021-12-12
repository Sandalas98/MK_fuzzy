from scipy.stats import shapiro, normaltest, mannwhitneyu


def check_normality(x, alpha=0.05, include_shapiro=True, include_dagostino=True):
    # H0 (p > alpha): data follows gauss distribution
    assert len(x) > 0
    stats = []

    if include_shapiro:
        shapiro_stat, shapiro_p = shapiro(x)
        stats.append(shapiro_p > alpha)

    if include_dagostino:
        dagostino_stat, dagostino_p = normaltest(x)
        stats.append(dagostino_p > alpha)

    return stats


def mann_whitney_test(x1, x2, alpha=0.05) -> bool:
    # nonparametric statistical significance test for determining
    # whether two independent samples were drawn from a population with the same distribution
    stat, p = mannwhitneyu(x1, x2)

    # returns True if distribution is the same for x1 and x2 (fail to reject H0)
    # False if distributions are different (reject H0)
    return p > alpha, p

