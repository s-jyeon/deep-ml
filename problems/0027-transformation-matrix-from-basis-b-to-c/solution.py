import numpy as np
import math 
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	B = np.array(B)
	C = np.array(C)

	C_inv = np.linalg.inv(C)
	P = C_inv @ B
	return list(P)


	
