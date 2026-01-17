import numpy as np

def compute_ltv(loan_amount, appraisals):
    """
    Compute LTV using multiple appraisals.
    Returns conservative, average, and worst-case LTV.
    """

    if loan_amount <= 0:
        return None

    # remove invalid appraisals
    appraisals = [a for a in appraisals if a and a > 0]

    if len(appraisals) == 0:
        return None

    appraisals = np.array(appraisals, dtype=float)

    conservative_value = np.percentile(appraisals, 25)
    average_value = appraisals.mean()
    worst_value = appraisals.min()

    return {
        "ltv_conservative": round(loan_amount / conservative_value, 3),
        "ltv_average": round(loan_amount / average_value, 3),
        "ltv_worst": round(loan_amount / worst_value, 3),
    }
