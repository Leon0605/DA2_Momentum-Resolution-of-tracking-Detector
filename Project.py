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



##### 4 Momentum Resolution #####

# Experiment Setup
n_before = 5        # number of detection planes before magnet
n_after = 3         # number of detection planes after magnet
delta_z = 20        # distance detector planes
width_cell = 0.5    # width of detection plane cells in 100 mikrometers (1mm is 1 then, 1cm is 10)
L = 100             # Magnet Lenght
B = 0.5             # strength of magnetic field in T


# Magnet class
class Magnet:
    def __init__(self, L, B):
        self.L = L
        self.B = B

    # generate change of true trajectory (WORK IN PROGRESS)
    def change(self, x_entry, s_entry):
        x_changed = x_entry
        s_changed = s_entry

    # interpolate change trajectory
    def change_interpolation(self, x_entry, s_entry, x_output, s_output, p_T, q, resolution=100):
        # generate interpolation points
        z_i = np.linspace(0, self.L+1, resolution)
        # z coordinates of midpoint of trajectory circle
        z_m = (x_output+self.L/s_output-x_entry)/(-s_entry**(-1)+s_output**(-1))
        # x coordinate of midpoint
        x_m = -s_entry**(-1)*z_m+x_entry
        # calculate angle roh
        roh = p_T/(q*self.B) * 1000
        # calculate interpolation
        interpolated_trajectory = np.sqrt(roh**2-(z_i-z_m)**2)+x_m
        return (interpolated_trajectory)
    
    # reconstruct p_T (WORK IN PROGRESS)
    def reconstruct_momentum(self, q, s_entry, s_output):
        theta = np.abs(s_entry - s_output)
        p_T_reconstructed = (self.L * self.B *q) / theta

        unc_p_T_reco = 0    # --> uncertainty von reco noch coden
        return (p_T_reconstructed, unc_p_T_reco)

# pull function
def pull(x_reco, x_gen, unc_x_reco):
    return (x_reco - x_gen)/unc_x_reco


### Warm-up Exercise
p_T_true = 0.3
magnet_10_0_5 = Magnet(L, B)

# a) Waiting for Code of Part 3 (WORK IN PROGRESS)

# b) Waiting for code of Part 3 but code for reconstruction done (WORK IN PROGRESS)
s_reco_before = 0
s_reco_after = 0
q = 0

p_T_reco = magnet_10_0_5.reconstruct_momentum(q, s_reco_before, s_reco_after)


### Estimate momentum resolution (vectorized with numpy)
## calculate histograms of differences
# prepare plots
fig, ax = plt.subplots(2, 8)
ax = ax.flatten()

# particle informations
s_reco_before_vec = 0
s_reco_after_vec = 0
q_vec = 0

# calculate reconstruction of momentum and the uncertainty
p_T_reco_vec, unc_vec = magnet_10_0_5.reconstruct_momentum(q, s_reco_before, s_reco_after)

# calculate differences between reco and true
p_T_diff = p_T_reco_vec - p_T_true

# calculate histogram, mean, std
hist, bins = np.histogram(p_T_diff, bins=50)
mean = np.mean(p_T_diff)
std = np.std(p_T_diff)

# plot histogram
ax[0].hist(p_T_reco_vec, bins=50, kde=True, color='skyblue', edgecolor='black')

## calculate histogram  of pulls



