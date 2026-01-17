def estimate_lgd(ltv, collateral_type="secured"):
    """
    Rule-based LGD estimation (Basel-style).
    """

    if ltv is None:
        return 0.60

    if collateral_type == "secured":
        if ltv < 0.5:
            return 0.20
        elif ltv < 0.7:
            return 0.35
        elif ltv < 0.9:
            return 0.50
        else:
            return 0.65

    # unsecured
    return 0.75
