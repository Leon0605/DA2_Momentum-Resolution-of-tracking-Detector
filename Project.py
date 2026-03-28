import numpy as np
import matplotlib as plt
import math
from sklearn.linear_model import LinearRegression


# Experiment Setup
n_before = 5        # number of detection planes before magnet
n_after = 3         # number of detection planes after magnet
delta_z = 20        # distance detector planes
cellWidth = 0.5    # width of detection plane cells in 100 mikrometers (1mm is 1 then, 1cm is 10)
L = 100             # Magnet Lenght
B = 0.5             # strength of magnetic field in T


class particle:

    def __init__(self, p_T):
        self.z0 = 0
        self.x0 = np.random.normal(0,1,None)
        # x position after the magnet
        self.xMagnet = self.x0
        self.s0 = np.random.normal(0,0.1,None)
        # slope after the magnet
        self.sMagnet = self.s0
        self.t0 = 0
        self.xactual = []
        self.layers = []
        # component of momentum transverse to magnetic field lines
        self.p_T = p_T
        # randomly assign particle charge
        self.q = np.random.choice([-1, 1])


def cellsHit(p, z, zMagnet=None):
    if zMagnet is None:
        x = p.x0 + p.s0 * (z) #calculate x position
    else:
        x = p.xMagnet + p.sMagnet * ((z - zMagnet)) #calculate x position after magnet
    p.xactual.append(x) 
    p.layers.append(np.floor(x/cellWidth)+1) #calculate and store cell index


def generatePoints(p, n):
    genX = []
    for i in range(0,n):
        genX.append(np.random.uniform(p.layers[i]-1*cellWidth, (p.layers[i])*cellWidth)) #generate data from hit cell index by drawing from cell interval as uniform distribution
    return np.array(genX)


##### 4 Momentum Resolution #####

# Magnet class
class Magnet:
    def __init__(self, L, B, zMagnet):
        self.L = L
        self.B = B
        self.z_beginn = zMagnet
        self.z_end = zMagnet+self.L

    # generate change of true trajectory
    def change(self, p):
        # modpint of circle
        rho = (p.p_T / p.q * self.B) / 1e-4
        norm = np.sqrt(1.0 + p.s0**2)
        M_x  = p.x0 + rho * (-(1.0 / norm) * p.q)
        M_z  = self.z_beginn    + rho * ((p.s0  / norm) * p.q)

        # xMagnet
        dz        = self.L - M_z
        x_straight = p.x0 + p.s0 * self.L             # Geradenverlängerung
        sqrt_disc  = np.sqrt(rho**2 - dz**2)
        x_cand     = [M_x + sqrt_disc, M_x - sqrt_disc]
        x_changed  = min(x_cand, key=lambda xc: abs(xc - x_straight))

        # sMagnet
        rx, rz = x_changed - M_x, self.L - M_z
        if p.q > 0:
            tx, tz = -rz,  rx
        else:
            tx, tz =  rz, -rx
        if tz < 0:
            tx, tz = -tx, -tz
        s_changed = tx / tz

        # change in particle
        p.xMagnet = x_changed
        p.sMagnet = s_changed

    # interpolate change trajectory
    def change_interpolation(self, x_entry, s_entry, p_T, q, resolution=100):
        z_mag_entry = n_before * delta_z
        z_i = np.linspace(z_mag_entry, z_mag_entry + self.L, resolution)

        rho = (p_T / q * self.B) / 1e-4
        norm = np.sqrt(1.0 + s_entry**2)
        M_x  = x_entry     + rho * (-(1.0 / norm) * q)
        M_z  = z_mag_entry + rho * ((s_entry / norm) * q)

        sign          = np.sign(x_entry - M_x)
        interpolated  = M_x + sign * np.sqrt(np.maximum(rho**2 - (z_i - M_z)**2, 0))
        return z_i, interpolated
    
    # reconstruct p_T (WORK IN PROGRESS)
    def reconstruct_momentum(self, q, s_entry, s_output):
        theta = np.abs(s_entry - s_output)  # calculated as approximation because small angle tan(x) = x
        p_T_reconstructed = (self.L * self.B *q) / theta * 1e-4

        unc_p_T_reco = 0    # --> uncertainty von reco noch coden
        return (p_T_reconstructed, unc_p_T_reco)


# pull function
def pull(x_reco, x_gen, unc_x_reco):
    return (x_reco - x_gen)/unc_x_reco


# Detector setup
z_detectors = np.array([i * delta_z for i in range(0, n_before+1)] + [delta_z*n_before+L + i * delta_z for i in range(0, n_after+1)])
z_0 = 0
zbegin_index = n_before
z_end = z_detectors[-1]


### Warm-up Exercise
p_T_true = 0.3
magnet_10_0_5 = Magnet(L, B, n_before*zbegin_index)

# a) Waiting for Code of Part 3 (WORK IN PROGRESS)
particle_warmup = particle(p_T_true)
magnet_10_0_5.change(particle_warmup) # calculate change because of magnet


# extrapolate trajectory 
cellsHit(particle_warmup, z_detectors[zbegin_index]) # calculate hits before magnets
cellsHit(particle_warmup, z_detectors[zbegin_index+1:], zMagnet=z_detectors[zbegin_index+1]) # calculate hits after magnet
curve_interpolation_warmup = magnet_10_0_5.change_interpolation(particle_warmup.xactual[zbegin_index], particle_warmup.s0, particle_warmup.xMagnet, particle_warmup.sMagnet, particle_warmup.p_T, particle_warmup.q)



## b) Waiting for code of Part 3 but code for reconstruction done (WORK IN PROGRESS)
#s_reco_before = 0
#s_reco_after = 0
#q = 0
#
#p_T_reco = magnet_10_0_5.reconstruct_momentum(q, s_reco_before, s_reco_after)
#
#
#### Estimate momentum resolution (vectorized with numpy)
### calculate histograms of differences
## prepare plots
#fig, ax = plt.subplots(2, 8)
#ax = ax.flatten()
#
## particle informations
#s_reco_before_vec = 0
#s_reco_after_vec = 0
#q_vec = 0
#
## calculate reconstruction of momentum and the uncertainty
#p_T_reco_vec, unc_vec = magnet_10_0_5.reconstruct_momentum(q, s_reco_before, s_reco_after)
#
## calculate differences between reco and true
#p_T_diff = p_T_reco_vec - p_T_true
#
## calculate histogram, mean, std
#hist, bins = np.histogram(p_T_diff, bins=50)
#mean = np.mean(p_T_diff)
#std = np.std(p_T_diff)
#
## plot histogram
#ax[0].hist(p_T_diff, bins=50, kde=True, color='skyblue', edgecolor='black')
#
### calculate histogram  of pulls
## prepare plots
#fig, ax = plt.subplots(2, 8)
#ax = ax.flatten()
#
## calculate pull
#p_T_pulls = pull(p_T_reco_vec, p_T_true, unc_vec)
#
## calculate histogram, mean, std
#hist, bins = np.histogram(p_T_pulls, bins=50)
#mean = np.mean(p_T_pulls)
#std = np.std(p_T_pulls)
#
## plot histogram
#ax[0].hist(p_T_pulls, bins=50, kde=True, color='skyblue', edgecolor='black')
#
#
#
#