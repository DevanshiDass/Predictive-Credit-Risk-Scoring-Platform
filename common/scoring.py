import numpy as np

# Scorecard configuration
PDO = 50                # Points to Double Odds
BASE_SCORE = 700        # Score at base odds
BASE_ODDS = 50          # (Good : Bad) odds at base score

FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

def pd_to_score(pd: float) -> int:
    # numerical safety
    pd = min(max(pd, 1e-6), 1 - 1e-6)
    odds = (1 - pd) / pd
    score = OFFSET + FACTOR * np.log(odds)
    return int(round(score))
