import numpy as np
import matplotlib as plt

class particle:

    def __init__(self, p_T):
        self.x0 = 0
        self.z0 = np.random.normal(0,10,None)
        self.s0 = np.random.normal(0,0.1,None)
        self.layers = None
        # component of momentum transverse to magnetic field lines
        self.p_T = p_T
        # randomly assign particle charge
        self.q = np.random.choice([-1, 1])

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


# Magnet class
class Magnet:
    def __init__(self, L, B):
        self.L = L
        self.B = B

    # generate change of true trajectory
    def change(self, x_entry, s_entry):
        x_changed = x_entry
        s_changed = s_entry

    # interpolate change trajectory (WORK IN PROGRESS)
    def change_interpolation(self, x_entry, s_entry, x_output, s_output):
        interpolated_trajectory = []
        return (interpolated_trajectory)
    
    # reconstruct p_T
    def reconstruct_momentum(self, q, s_entry, s_output):
        theta = np.abs(s_entry - s_output)
        p_T_reconstructed = (self.L * self.B *q) / theta
        return p_T_reconstructed


# Warm-up Exercise

