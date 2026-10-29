import math
import numpy as np
def softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    softmax = []

    for score in scores:
        res = np.exp(score-max_score) / sum(np.exp(score-max_score)for score in scores)
        softmax.append(res)

    return softmax

  

