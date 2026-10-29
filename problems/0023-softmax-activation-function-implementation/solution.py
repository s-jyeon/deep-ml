import math
import numpy as np
def softmax(scores: list[float]) -> list[float]:
    score_arr = np.array(scores)
    exp_score = np.exp(score_arr-np.max(score_arr))

    return exp_score/sum(exp_score)

  

