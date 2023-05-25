from anglesOnlyRIOD import pseudo_inverse_RIOD
import numpy as np 
from STMint.STMint import (
    STMint,
)  # uses STMint package (https://github.com/SIOSlab/STMInt)

# Use STMInt to generate observations for example purposes (also provides STMs and STTs)

num_observations = 10
#using a nondimensional circular orbit
#[x,y,z,vx,vy,vz]
chief_orbit = np.array([1, 0, 0, 0, 1, 0])
#small offset along-track
deputy_orbit = chief_orbit + np.array([0, 0.001, 0, 0, 0, 0])

integ = STMint(preset="twoBody", variational_order=2)

# simulating chief orbit (e.g. earth)
[chief_states, STMs, STTs, chief_times] = integ.dynVar_int2(
    [0, (2 * np.pi)],
    chief_orbit,
    output="all",
    max_step=0.01,
    method="DOP853",
    t_eval=np.linspace(0, 2 * np.pi, num_observations),
)

# simulated deputy orbit (e.g. satellite)
deputy = integ.dyn_int(
    [0, (2 * np.pi)],
    deputy_orbit,
    max_step=0.01,
    method="DOP853",
    t_eval=np.linspace(0, 2 * np.pi, num_observations),
)

deputy_states = deputy.y
deputy_times = deputy.t

# We can also calculate the relative observations, the STMs, and the STTs.
STMs = np.array(STMs)[1:, 0:3, :]  # remove derivates in STMs and STTs
STTs = np.array(STTs)[1:, 0:3, :, :]
line_of_sight = (deputy_states.T - chief_states)[1:, 0:3] # compute the line of sight vectors

# use function defined in angles-only-ROID.py
determined_delta = pseudo_inverse_RIOD(line_of_sight, STMs, STTs)

print("Pseudo-inverse eigenvalue algorithm results:", determined_delta)
print("True Solution: ", deputy_orbit - chief_orbit)