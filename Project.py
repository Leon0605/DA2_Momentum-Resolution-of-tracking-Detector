import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import optimize
from sklearn.linear_model import LinearRegression


# Experiment Setup
n_before = 5        # number of detection planes before magnet
n_after = 3         # number of detection planes after magnet
dZ = 20             # distance detector planes (2cm = 20mm)
cellWidth = 0.5     # width of detection plane cells in 100 mikrometers (1mm is 1 then, 1cm is 10)
L = 100             # Magnet Lenght
B = 0.5             # strength of magnetic field in T


# particle class
class particle:

    def __init__(self, p_T):
        self.z0 = 0
        self.x0 = np.random.normal(0,1,None)
        self.x0reco = 0
        self.x0recoUncert = 0
        # x position after the magnet
        self.xMagnet = self.x0
        self.s0 = np.tan(np.random.normal(0,0.1,None))
        self.s0reco = 0
        self.s0recoUncert = 0
        # slope after the magnet
        self.sMagnet = self.s0
        self.t0 = 0
        self.xactual = []
        self.layers = []
        # component of momentum transverse to magnetic field lines
        self.p_T = p_T
        # randomly assign particle charge
        self.q = np.random.choice([-1, 1])


# Magnet class
class Magnet:
    def __init__(self, L, B, zMagnet):
        self.L = L
        self.B = B
        self.z_beginn = zMagnet
        self.z_end = zMagnet+self.L

    # generate change of true trajectory
    def change(self, p):
        # p_T in GeV, B in T → rho in m, *1000 für mm
        # sign of rho defines curvature (q > 0 → rho > 0)
        rho = p.p_T / (0.3 * p.q * self.B) * 1000   # signed, in mm

        # entry at beginning of magnet
        x_entry = p.x0 + p.s0 * self.z_beginn
        z_entry = self.z_beginn
        alpha0  = np.arctan(p.s0)

        # calculate center
        # orthogonal-left towards momentumvector (sin α₀, cos α₀) is (-cos α₀, sin α₀)
        # M lays in that direction in distance rho:
        #   q > 0 → rho > 0 → centre left  → CCW-bahn (-x)
        #   q < 0 → rho < 0 → centre rechts → CW-bahn  (+x)
        M_x = x_entry + rho * (-np.cos(alpha0)) 
        M_z = z_entry + rho *   np.sin(alpha0)

        # calculate output x
        z_exit = self.z_end
        discriminant = rho**2 - (z_exit - M_z)**2
        p.xMagnet = M_x + np.sign(rho) * np.sqrt(max(discriminant, 0.0))

        # -calculate output s
        # Tangente of circle at output: slope = -(z_exit - M_z) / (x_exit - M_x)
        p.sMagnet = -(z_exit - M_z) / (p.xMagnet - M_x)

    # interpolate change trajectory
    def change_interpolation(self, x_entry, s_entry, p_T, q, resolution=100):
        z_i = np.linspace(self.z_beginn, self.z_beginn + self.L, resolution)

        rho = p_T / (0.3 * q * self.B) * 1000

        norm = np.sqrt(1.0 + s_entry**2)
        M_x  = x_entry     + rho * (-1.0 / norm)
        M_z  = self.z_beginn + rho * (s_entry / norm)

        interpolated = M_x + np.sign(rho) * np.sqrt(np.maximum(rho**2 - (z_i - M_z)**2, 0))
        return z_i, interpolated

    # reconstruct p_T
    def reconstruct_momentum(self, q, s_entry, s_output, unc_s_entry, unc_s_output, covariance_s_in_out=0):
        theta = np.abs(np.arctan(s_entry) - np.arctan(s_output))
        p_T_reconstructed = (self.L * self.B * np.abs(q)) / theta * 3e-4

        # calculate error propagation
        delta_angle = np.arctan(s_entry) - np.arctan(s_output)
        unc_p_T_reco = np.sqrt((self.L * q * self.B * 3e-4)**2 * (unc_s_entry**2 / ((1+s_entry**2)**2 * delta_angle**4) + unc_s_output**2 / ((1+s_output**2)**2 * delta_angle**4) - 2 * covariance_s_in_out / ((1+s_entry**2) * (1+s_output**2) * delta_angle**4)))
        return (p_T_reconstructed, unc_p_T_reco)


#calculate and store cell index
def cellsHit(p, z, zMagnet=None):
    if zMagnet is None:
        x = p.x0 + p.s0 * (z) #calculate x position
    else:
        x = p.xMagnet + p.sMagnet * ((z - zMagnet)) #calculate x position after magnet
    p.xactual.append(x)
    if(x >= 0):
        p.layers.append(np.floor(x/cellWidth)+1)
    else:
        p.layers.append(np.floor(x/cellWidth)) #calculate and store cell index


def generatePoints(p, n):
    genX = []
    for i in range(0,n):
        genX.append(np.random.uniform(p.layers[i]-1*cellWidth, (p.layers[i])*cellWidth)) #generate data from hit cell index by drawing from cell interval as uniform distribution
    return np.array(genX)


def generatePoints(p):
    #for i in range(0,n):
        #genX.append(np.random.uniform(p.layers[i]-1*cellWidth, (p.layers[i])*cellWidth)) #generate data from hit cell index by drawing from cell interval as uniform distribution
    genX = [(l-1)*cellWidth + cellWidth/2 if l > 0 else l*cellWidth + cellWidth/2 for l in p.layers]
    uncertainties = [cellWidth/np.sqrt(12) for _ in range(0,len(p.layers))]
    return np.array(genX), np.array(uncertainties)


def line(z, slope, intercept):
    return slope * z + intercept


def run(p1):
    Z = []
    for i in range(0, n_before):
        z = (i+1)*dZ
        Z.append(z)
        cellsHit(p1, z)
    X, uncertainties = generatePoints(p1)
    #coeffs, cov = np.polyfit(np.array(Z), X, 1, cov=True) #cov = covariance matrix of fitted parameter
    coeffs, cov = optimize.curve_fit(line, np.array(Z), X, sigma=uncertainties, absolute_sigma=True)
    p1.s0reco, p1.x0reco = coeffs
    p1.s0recoUncert, p1.x0recoUncert = np.sqrt(np.diag(cov))
    pulls0, pullx0 = pull(p1)
    #plot_trajectories(p1, X, Z, uncertainties)
    return pulls0, pullx0

def plot_trajectories(p1, X, Z, uncertainties):
    z_line = np.linspace(0, n_before*dZ, 10)
    x_true_line = p1.x0 + p1.s0 * z_line
    x_reco_line = p1.x0reco + p1.s0reco * z_line

    plt.figure()
    plt.plot(z_line, x_true_line, '--', label='True trajectory')
    plt.errorbar(Z, X, yerr=uncertainties, fmt='.', label='Hit positions')
    plt.plot(z_line, x_reco_line, '-', label='Reconstructed trajectory')
    plt.xlabel('z')
    plt.ylabel('x')
    plt.legend()
    plt.show()


def simulation():
    s0Residuals, x0Residuals = [],[]
    pulls0, pullx0 = [], []
    for _ in range(0, 1000):
        p = particle()
        ps0, px0 = run(p)
        s0Residuals.append(p.s0reco-p.s0)
        x0Residuals.append(p.x0reco-p.x0)
        pulls0.append(ps0)
        pullx0.append(px0)
    s0Residuals = np.array(s0Residuals)
    x0Residuals = np.array(x0Residuals)
    pulls0 = np.array(pulls0)
    pullx0 = np.array(pullx0)

    s0ResidualsMean = np.mean(s0Residuals)
    s0ResidualsSD = np.std(s0Residuals)
    x0ResidualsMean = np.mean(x0Residuals)
    x0ResidualsSD = np.std(x0Residuals)
    print(s0ResidualsMean, s0ResidualsSD, x0ResidualsMean, x0ResidualsSD)

    pulls0Mean = np.mean(pulls0)
    pulls0SD= np.std(pulls0)

    pullx0Mean = np.mean(pullx0)
    pullx0SD = np.std(pullx0)

    print(pulls0Mean, pulls0SD, pullx0Mean, pullx0SD)

    plot_histograms(pulls0, "Pull S0")
    plot_histograms(pullx0, "Pull X0")
    plot_histograms(s0Residuals, "S0 Residuals")
    plot_histograms(x0Residuals, "X0 Residuals")

def plot_histograms(residuals, title):
    plt.figure()
    plt.hist(residuals, bins=50)
    plt.title(title)
    plt.show()

def pull(p):
    pulls0 = (p.s0reco - p.s0) / p.s0recoUncert
    pullx0 = (p.x0reco - p.x0) / p.x0recoUncert
    return pulls0, pullx0
#simulation()


##### 4 Momentum Resolution #####

# pull function
def pull(x_reco, x_gen, unc_x_reco):
    return (x_reco - x_gen)/unc_x_reco

# function to run Part 4 Momentum Resolution
def Momentum_Resolution():
    # Detector setup
    z_detectors = np.array([i * dZ for i in range(0, n_before+1)] + [dZ*n_before+L + i * dZ for i in range(0, n_after+1)])
    z_0 = 0
    zbegin_index = n_before
    z_end = z_detectors[-1]

    print(f"detector setup (z-coord): {z_detectors}")
    print()


    ### Warm-up Exercise
    p_T_true = 0.3
    magnet_10_0_5 = Magnet(L, B, n_before*dZ)

    # a) Waiting for Code of Part 3 (Work in PROGRESS)
    particle_warmup = particle(p_T_true)
    magnet_10_0_5.change(particle_warmup) # calculate change because of magnet


    # extrapolate trajectory 
    for z in z_detectors[:zbegin_index+1]:    # before magnet
        cellsHit(particle_warmup, z) # calculate hits before magnets

    for z in z_detectors[zbegin_index+1:]:
        cellsHit(particle_warmup, z, zMagnet=magnet_10_0_5.z_end) # calculate hits after magnet

    print(f"slope before magnet: {particle_warmup.s0}")
    print(f"slope after magnet: {particle_warmup.sMagnet}")
    print(f"x-coord beginning: {particle_warmup.x0}")
    print(f"x-coord before magnet: {particle_warmup.xactual[zbegin_index]}")
    print(f"x-coord after magnet: {particle_warmup.xMagnet}")
    print()

    print(particle_warmup.q)

    print(f"hitpoints before magnet: {particle_warmup.xactual[:zbegin_index]}")
    print(f"hitpoints after magnet: {particle_warmup.xactual[zbegin_index+1:]}")
    print()
    print(f"hitpoints: {particle_warmup.xactual}")
    print(f"cell index: {particle_warmup.xactual}")
    print()

    # calculat trejectory through magnet
    curve_z_i, curve_interpolation_warmup = magnet_10_0_5.change_interpolation(particle_warmup.xactual[zbegin_index], particle_warmup.s0, p_T_true, particle_warmup.q)
    
    # plot of simulation of part 4a
    plt.figure()

    for z in z_detectors:
        plt.plot([z, z + 10.0**(-6)], [min(particle_warmup.xactual)-10, max(particle_warmup.xactual)+10], color="lightblue")
    plt.plot(z_detectors[:zbegin_index+1], particle_warmup.xactual[:zbegin_index+1], label="True Trajectory before Magnet")
    plt.plot(z_detectors[zbegin_index+1:], particle_warmup.xactual[zbegin_index+1:], label="True Trajectory after Magnet")
    plt.plot(curve_z_i, curve_interpolation_warmup, label="INterpolated Trajectory through Magnet")
    plt.xlabel("z")
    plt.ylabel("x")
    plt.legend()
    plt.show()



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
    ax[0].hist(p_T_diff, bins=50, kde=True, color='skyblue', edgecolor='black')

    ## calculate histogram  of pulls
    # prepare plots
    fig, ax = plt.subplots(2, 8)
    ax = ax.flatten()

    # calculate pull
    p_T_pulls = pull(p_T_reco_vec, p_T_true, unc_vec)

    # calculate histogram, mean, std
    hist, bins = np.histogram(p_T_pulls, bins=50)
    mean = np.mean(p_T_pulls)
    std = np.std(p_T_pulls)

    # plot histogram
    ax[0].hist(p_T_pulls, bins=50, kde=True, color='skyblue', edgecolor='black')

# run Part 4 Momentum Resolution
Momentum_Resolution()