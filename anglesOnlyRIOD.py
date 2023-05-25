# Main Functions for Angles Only Relative Orbit Determination
import numpy as np
from scipy.linalg import eig

# Pseudo-inverse eigenvalue algorithm to obtain initial guess at passive angles only relative orbit determniation algorithm.
def pseudo_inverse_RIOD(observations, STMs, STTs):

    # normalize line of sight vectors
    for i in range(len(observations)):
        observations[i] = observations[i]/np.linalg.norm(observations[i])

    A = build_a_matrix(observations, STMs)

    # generate initial guess
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    initial_guess = vh[5] * np.sign(observations[0] @ STMs[0] @ vh[5])

    # generate B matrix
    B = build_b_matrix(observations, STTs, initial_guess)

    least_squares_sol = np.linalg.pinv(A) @ B

    # find eigenvalues and eigenvectors of least squares solution
    eig_vals, eig_vecs = eig(least_squares_sol)
    # tranpose eigenvectors
    eig_vecs = eig_vecs.T

    # find eigenvector with eigenvalue 
    # Select eignpair that mazimizes |x_hat \cdot x_0 (initial guess)|
    dot_max = -1
    index = 0
    for i in range(len(eig_vals)):
        dot_prod = abs(np.dot(eig_vecs[i], initial_guess))
        if dot_prod > dot_max:
            dot_max = dot_prod
            index = i

    # Find eignpair
    eig_val_max = eig_vals[index]
    eig_vec_max = eig_vecs[index]

    # Find x_hat
    delta_x0 = np.real((-2. * eig_vec_max)/eig_val_max)
    delta_x0 = delta_x0 * np.sign(observations[0] @ STMs[0] @ delta_x0)

    return delta_x0

# Quadratic eigenvalue based orbit determination method presented in the paper (see alg 2)
def quadratic_eigenvalue_RIOD(observations, STMs, STTs):
    # normalize line of sight vectors
    for i in range(len(observations)):
        observations[i] = observations[i] / np.linalg.norm(observations[i])

    # Build matrix A detailed in the paper
    A = build_a_matrix(observations, STMs)
  
    # Compute SVD of A, and find initial guess
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    initial_guess = vh[5]

    # Build matrix B detailed in the paper
    B = build_b_matrix(observations, STTs, initial_guess)

    # Generate Q,L,C for solving general quadratic eigenvalue problem
    Q = np.matmul(B.T, B)
    L = -np.matmul(B.T, A) - np.matmul(A.T, B)
    C = np.matmul(A.T, A)

    # Create block matrix for general quadratic eigenvalue problem
    M = np.block([[np.zeros(C.shape), C],
               [C, L]])
    N = np.block([[C, np.zeros( (C.shape[0], Q.shape[1]))],     
                [np.zeros( (Q.shape[0], C.shape[1] ) ), -Q]])

    # Solve quadratic eigenvalue problem
    eig_vals, eig_vecs = eig(M, N)

    eig_vecs = eig_vecs.T[:, 0 : int(len(eig_vecs) / 2)]

    # Normalize eigenvectors
    for i in range(len(eig_vecs)):
        eig_vecs[i] = eig_vecs[i] / np.linalg.norm(eig_vecs[i])

    # Select eigenpair that mazimizes |x_hat \cdot x_0 (initial guess)|
    dot_max = -1
    index = 0
    for i in range(len(eig_vals)):
        dot_prod = abs(np.dot(eig_vecs[i], initial_guess))
        if dot_prod > dot_max:
            dot_max = dot_prod
            index = i

    # Find eigenpair
    eig_val_max = eig_vals[index]
    eig_vec_max = eig_vecs[index]


    # Compute final solution (wth sign adjustment):
    deltaX_0 = np.real(-2 * eig_vec_max * eig_val_max)
    deltaX_0 = deltaX_0 * np.sign(observations[0] @ STMs[0] @ deltaX_0)

    return deltaX_0

# Build matrix A detailed in the paper
def build_a_matrix(los, STMs):
    l0 = genSkew(los[0][0], los[0][1], los[0][2])
    A = np.matmul(l0, STMs[0])

    for i in range(1,len(los)):
        l = genSkew(los[i][0], los[i][1], los[i][2])
        prod = np.matmul(l, STMs[i])
        A = np.append(A, prod, axis=0)

    return A

# Build matrix B detailed in the paper
def build_b_matrix(los, STTs, initialGuess):
    B = np.einsum('im,mjp,p -> ij', genSkew(los[0][0], los[0][1], los[0][2]), STTs[0], initialGuess)
    for i in range(1,len(los)):
        l = genSkew(los[i][0], los[i][1], los[i][2])
        B = np.append(B, np.einsum('im,mjp,p -> ij', l, STTs[i], initialGuess), axis=0)
    return B
# define a skew function, to generate skew matrix from vector
def genSkew(x, y, z):
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])