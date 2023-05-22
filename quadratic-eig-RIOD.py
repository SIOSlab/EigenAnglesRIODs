# Pseudo-inverse eigenvalue algorthim to obtain initial guess at passive angles only relative orbit determniation algorithm.

from STMint.STMint import (
    STMint,
)  # uses STMint package (https://github.com/SIOSlab/STMInt)
import numpy as np
from scipy.linalg import eig


# define a skew function, to generate skew matrix from vector
def genSkew(x, y, z):
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


# Generate observations for example purposes
observations = 6
chiefOrbit = np.array([1, 0, 0, 0, 1, 0])
deputyOrbit = chiefOrbit + np.array([0, 0.001, 0, 0, 0, 0])
deputyDelta = deputyOrbit - chiefOrbit
# Generate observations (STTs, STMs)
integ = STMint(preset="twoBody", variational_order=2)

# simulating chief orbit (e.g. earth)
[chiefStates, STMs, STTs, chiefTimes] = integ.dynVar_int2(
    [0, (2 * np.pi)],
    chiefOrbit,
    output="all",
    max_step=0.01,
    method="DOP853",
    t_eval=np.linspace(0, 2 * np.pi, observations),
)

# simulated deputy orbit (e.g. satellite)
deputy = integ.dyn_int(
    [0, (2 * np.pi)],
    deputyOrbit,
    max_step=0.01,
    method="DOP853",
    t_eval=np.linspace(0, 2 * np.pi, observations),
)

deputyStates = deputy.y
deputyTimes = deputy.t

# We can calculate the relative observations, the STMs, and the STTs.
STMs = np.array(STMs)[1:, 0:3, :]  # remove derivates in STMs and STTs
STTs = np.array(STTs)[1:, 0:3, :, :]
line_of_sight = (deputyStates.T - chiefStates)[1:, 0:3]

# normalize line of sight vectors
for i in range(len(line_of_sight)):
    line_of_sight[i] = line_of_sight[i] / np.linalg.norm(line_of_sight[i])

# Generating A matrix
l0 = genSkew(line_of_sight[0][0], line_of_sight[0][1], line_of_sight[0][2])
A = np.matmul(l0, STMs[0])

for i in range(1, len(line_of_sight)):
    l = genSkew(line_of_sight[i][0], line_of_sight[i][1], line_of_sight[i][2])
    prod = np.matmul(l, STMs[i])
    A = np.append(A, prod, axis=0)

u, s, vh = np.linalg.svd(A, full_matrices=True)
initialGuess = vh[5]

# Generating B matrix

B = np.einsum("im,mjp,p -> ij", l0, STTs[0], vh[5])
for i in range(1, len(line_of_sight)):
    l = genSkew(line_of_sight[i][0], line_of_sight[i][1], line_of_sight[i][2])
    B = np.append(B, np.einsum("im,mjp,p -> ij", l, STTs[i], vh[5]), axis=0)

# We can now form the generalized linear eignvalue problem:
Q = np.matmul(B.T, B)
L = -np.matmul(B.T, A) - np.matmul(A.T, B)
C = np.matmul(A.T, A)

# We can now create the matrices:
M = np.block([[np.zeros(C.shape), C],
               [C, L]])
N = np.block([[C, np.zeros( (C.shape[0], Q.shape[1]))], 
              [np.zeros( (Q.shape[0], C.shape[1] ) ), -Q]])



# Solving this generalized eigenvalue problem:
eigVals, eigVecs = eig(M, N)

# We then remove the bottom half of the eigenvalues:
eigVecs = eigVecs.T[:, 0 : int(len(eigVecs) / 2)]

# Renormalizing the eigenvectors:
for i in range(len(eigVecs)):
    eigVecs[i] = eigVecs[i] / np.linalg.norm(eigVecs[i])

# Select eignpair that mazimizes |x_hat \cdot x_0 (initial guess)|
dotMax = -1
index = 0
for i in range(len(eigVals)):
    dotProd = abs(np.dot(eigVecs[i], initialGuess))
    if dotProd > dotMax:
        dotMax = dotProd
        index = i

# Find corresponding eigenpair:
eigVal = eigVals[index]
eigVec = eigVecs[index]

# Compute final solution (wth sign adjustment):

deltaX_0 = np.real(-2. * eigVec * eigVal)
deltaX_0 = deltaX_0 * np.sign(line_of_sight[0] @ STMs[0] @ deltaX_0)

print("Final Solution: ", deltaX_0)
print("True Solution: ", deputyDelta)