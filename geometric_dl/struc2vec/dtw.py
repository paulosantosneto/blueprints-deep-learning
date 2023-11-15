import numpy as np

def dynamic_time_warping(A: np.array, B: np.array):
    
    A_size = len(A)
    B_size = len(B)

    # --- Cost Matrix ---
    
    M = np.zeros([A_size+1, B_size+1], dtype=float)
    M[A_size, :], M[: , 0] = np.PINF, np.PINF
    M[A_size, 0] = 0
    
    for i in range(A_size -1, -1, -1):
        for j in range(1, B_size + 1):

            mins = np.array([M[i+1, j-1], M[i, j-1], M[i+1, j]])
            dij = (max(A[A_size-i-1], B[j-1]) / (min(A[A_size-i-1], B[j-1]))) - 1
            M[i, j] = dij + mins[np.isfinite(mins)].min()
    
    # --- Warping Path ---
    
    row_idx, col_idx = 0, B_size
    
    path = [M[row_idx, col_idx]]

    while (row_idx != A_size and col_idx != 1):
        
        mins = np.array([M[row_idx+1, col_idx-1], M[row_idx, col_idx-1], M[row_idx+1, col_idx]])
        smaller = mins[np.isfinite(mins)].min()
        
        path.append(smaller)

        if M[row_idx+1, col_idx-1] == smaller:
            row_idx += 1
            col_idx -= 1
        elif M[row_idx, col_idx-1] == smaller:
            col_idx -= 1
        else:
            row_idx += 1

    return np.array(path).mean()
