import numpy as np
import matplotlib as plt

class particle:

    def __init__(self):
        self.x0 = 0
        self.z0 = np.random.normal(0,10,None)
        self.s0 = np.random.normal(0,0.1,None)
        self.layers = None

def cellsHit(dZ,n,z):
    pass



### 4 Momentum Resolution

# Experiment Setup
n_before = 5    # number of detection planes before magnet
n_after = 3     # number of detection planes after magnet
delta_z = 2     # distance detector planes
width_cell = 5  # width of detection plane cells in 100 mikrometers (1mm is 10 then, 1cm is 100)
L = 1000        # Magnet Lenght
B = 0.5         # strength of magnetic field in T


# Warm-up Exercise

