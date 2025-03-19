import math
import numpy as np

#Shannon formula
def unconditional_entropy_from_array(arr, do_normilize: bool = True):
    def normalize_array(arr):
        norm = []
        total = sum(arr)
        for x in arr:
            val = 0
            if (x != 0 and total != 0):
                val = x / total
            norm.append(val)
        return norm
    
    normalized_arr = arr
    if (do_normilize):
        normalized_arr = normalize_array(arr)

    ntrp = 0
    for f in normalized_arr:
        val = 0
        if f != 0:
            val = f * math.log2(f) 
        ntrp -= val
    return ntrp

def unconditional_entropy_from_matrix(matrix: np.ndarray, do_normilize: bool = True):
    matrix = matrix.flatten()
    return unconditional_entropy_from_array(matrix, do_normilize)


def partial_conditional_entropy_from_matrix(matrix: np.ndarray, y_index: int, do_normilize: bool = True):
    row = matrix[y_index, :]
    return unconditional_entropy_from_array(row, do_normilize)


def total_conditional_entropy_from_matrix(matrix: np.ndarray, do_normilize: bool = True):
    total_entropy = 0.0
    for y_index in range(matrix.shape[0]):
        p_y = np.sum(matrix[y_index, :])
        if p_y > 0:
            total_entropy += p_y * partial_conditional_entropy_from_matrix(matrix, y_index, do_normilize)
    
    return total_entropy