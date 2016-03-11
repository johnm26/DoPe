from numpy import pi, sqrt

#####################
# PENDULUM PARAMETERS
#####################
# Arm lengths and bob masses.
l1 = 1.0
l2 = 1.0
m1 = 1.0
m2 = 1.0

# Initial conditions.
t0 = 180 * pi / 180.0
p0 = 1 * pi/ 180.0
td0 = 0.0
pd0 = 0.0

#####################
# ENSEMBLE PARAMETERS
#####################
# Size of blob of initial conditions in theta/phi space.
# blobt0/blobp0 is the center of the blob in the theta/phi plane.
# blobdt/blobdp are side-lengths of the blob in theta/phi.
# blobtd0/blobpd0 are initial velocities of pendula in the blob.
# blobnt/blobnp are the number of samples in theta/phi to take.
blobt0 = pi
blobp0 = 1 * pi / 180.0
blobdt = 0.5 * pi / 180.0
blobdp = 0.5 * pi / 180.0
blobtd0 = 0.0
blobpd0 = 0.0
blobnt = 5
blobnp = 5

###############################
# SOLUTION ALGORITHM PARAMETERS
###############################
# Timestep and number of steps to take.
dt = 0.1
n  = 500