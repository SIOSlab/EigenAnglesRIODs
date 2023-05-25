# Overdetermined Eigenvector Approach to Passive Angles-Only Relative Orbit Determination
#EigenAnglesRIOD

# File Contents:
- `anglesOnlyRIOD.py` contains the main functions for both algorithms. Specfically, it contains functions to create A and B matricies constructed in the paper. Further, it implements algorithm 1 (pseudoinverse) and algorithm 2 (quadratic eigenvalue) from the paper.
- `quadratic-eig-example.py` show how to use the quadratic eigenvalue algorithm function to solve the relative orbit determination problem.
- `pseudo-inverse-example.py` show how to use the pseudoinverse algorithm to solve the relative orbit determination problem.

# Required Packages:
- `STMint` which can be downloaded at https://github.com/SIOSlab/STMInt
- `numpy`
- `scipy`