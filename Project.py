import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
from scipy.interpolate import CubicHermiteSpline
from scipy.stats import gaussian_kde, norm
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
        # x position at beginning of detector
        self.z0 = 0
        self.x0 = np.random.normal(0,1,None)
        self.x0reco = 0
        self.x0recoUncert = 0
        # x position after the magnet
        self.xMagnet = self.x0
        self.xMagnetreco = 0
        self.xMagnetrecoUncert = 0
        # slope before magnet
        self.s0 = np.tan(np.random.normal(0,0.1,None))
        self.s0reco = 0
        self.s0recoUncert = 0
        # slope after the magnet
        self.sMagnet = self.s0
        self.sMagnetreco = 0
        self.sMagnetrecoUncert = 0
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
        p.xMagnet = M_x + np.sign(rho) * np.sqrt(np.maximum(discriminant, 0.0))

        # -calculate output s
        # Tangente of circle at output: slope = -(z_exit - M_z) / (x_exit - M_x)
        p.sMagnet = -(z_exit - M_z) / (p.xMagnet - M_x)
        return M_z, M_x

    # interpolate change trajectory
    def change_interpolation_1(self, x_entry, s_entry, p_T, q, resolution=100):
        z_i = np.linspace(self.z_beginn, self.z_beginn + self.L, resolution)

        rho = p_T / (0.3 * q * self.B) * 1000

        nor = np.sqrt(1.0 + s_entry**2)
        M_x  = x_entry     + rho * (-1.0 / nor)
        M_z  = self.z_beginn + rho * (s_entry / nor)

        interpolated = M_x + np.sign(rho) * np.sqrt(np.maximum(rho**2 - (z_i - M_z)**2, 0))
        return z_i, interpolated
    
    # interpolate change trajectory
    def change_interpolation_2(self, x_entry, s_entry, x_exit, s_exit, resolution=100):
        z_i = np.linspace(self.z_beginn, self.z_beginn + self.L, resolution)

        z_endpoints = np.array([self.z_beginn, self.z_beginn + self.L])
        x_endpoints = np.array([x_entry, x_exit])
        slopes      = np.array([s_entry, s_exit])

        spline = CubicHermiteSpline(z_endpoints, x_endpoints, slopes)
        return z_i, spline(z_i)

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


    ### Warm-up Exercise ###
    p_T_true = 0.3
    magnet_10_0_5 = Magnet(L, B, n_before*dZ)


    ## a) Extrapolate trajectory and generate hitpoints with uncertainties ##
    particle_warmup = particle(p_T_true)
    M_z, M_x = magnet_10_0_5.change(particle_warmup) # calculate change because of magnet

    # extrapolate trajectory 
    for z in z_detectors[:zbegin_index+1]:    # before magnet
        cellsHit(particle_warmup, z) # calculate hits before magnets

    for z in z_detectors[zbegin_index+1:]:
        cellsHit(particle_warmup, z, zMagnet=magnet_10_0_5.z_end) # calculate hits after magnet

    # calculate trajectory through magnet
    curve_z_i, curve_interpolation_warmup = magnet_10_0_5.change_interpolation_1(particle_warmup.xactual[zbegin_index], particle_warmup.s0, p_T_true, particle_warmup.q)
    
    # calculate hitposition and uncertainty
    hits_warmup, unc_warmup = generatePoints(particle_warmup)


    ## b) Fit straight lines and calculate reco p_T ##
    # calculate linear regression before magnet
    coeffs_before, cov_before = optimize.curve_fit(line, z_detectors[:zbegin_index+1], hits_warmup[:zbegin_index+1], sigma=unc_warmup[:zbegin_index+1], absolute_sigma=True)
    particle_warmup.s0reco, particle_warmup.x0reco = coeffs_before
    particle_warmup.s0recoUncert, particle_warmup.x0recoUncert = np.sqrt(np.diag(cov_before))

    # calculate linear regression after magnet
    coeffs_after, cov_after = optimize.curve_fit(line, z_detectors[zbegin_index+1:], hits_warmup[zbegin_index+1:], sigma=unc_warmup[zbegin_index+1:], absolute_sigma=True)
    particle_warmup.sMagnetreco, intercept_after_reco = coeffs_after
    particle_warmup.xMagnetreco = line(magnet_10_0_5.z_end, particle_warmup.sMagnetreco, intercept_after_reco)
    unc_slope, unc_intercept = np.sqrt(np.diag(cov_after))
    particle_warmup.sMagnetrecoUncert = unc_slope
    particle_warmup.xMagnetrecoUncert = np.sqrt((magnet_10_0_5.z_end * unc_slope)**2 + unc_intercept**2)

    # calculate line before magnet
    x_line_reco_before_warmup = line(z_detectors[:zbegin_index+1], particle_warmup.s0reco, particle_warmup.x0reco)

    # calculate line after magnet
    x_line_reco_after_warmup = line(z_detectors[zbegin_index+1:], particle_warmup.sMagnetreco, particle_warmup.xMagnetreco - particle_warmup.sMagnetreco * magnet_10_0_5.z_end)
    
    # calculate reconstructed trajectory through magnet
    curve_z_i, curve_interpolation_reco_warmup = magnet_10_0_5.change_interpolation_2(x_line_reco_before_warmup[zbegin_index], particle_warmup.s0reco, particle_warmup.xMagnetreco, particle_warmup.sMagnetreco)

    # calculate reconstruction of p_T
    p_T_reco_warmup, p_T_reco_unc_warmup = magnet_10_0_5.reconstruct_momentum(particle_warmup.q, particle_warmup.s0reco, particle_warmup.sMagnetreco, particle_warmup.s0recoUncert, particle_warmup.sMagnetrecoUncert)
    
    print(f"--- Warmup: Reconstruction of p_T ---")
    print(f"p_T true: {p_T_true} GeV")
    print(f"p_T reconstructed: {p_T_reco_warmup:.5f} GeV")
    print(f"diff p_T true and reco: {np.abs(p_T_true - p_T_reco_warmup):.5f} GeV")
    print(f"uncertainty of reconstruction: {p_T_reco_unc_warmup:.5f} GeV")
    print()

    # plot of simulation
    plt.figure(figsize=(20,12))

    # plot detector layers
    for z in z_detectors:
        plt.plot([z, z + 10.0**(-6)], [np.min(hits_warmup)-0.5, np.max(hits_warmup)+0.5], linewidth=1, color="lightgrey")

    # plot reconstructed trajectory
    #plt.scatter(z_detectors, hits_warmup, marker="x", color="black", s=0.5, label='Uncertainties Hit positions')
    plt.plot(z_detectors[:zbegin_index+1], x_line_reco_before_warmup, color="lightblue", label='Reconstructed Trajectory')
    plt.plot(z_detectors[zbegin_index+1:], x_line_reco_after_warmup, color="lightblue")
    plt.plot(curve_z_i, curve_interpolation_reco_warmup, color="lightblue")
    # plot true trajectory
    plt.plot(z_detectors[:zbegin_index+1], particle_warmup.xactual[:zbegin_index+1],linestyle="dashed", color="orange", label="True Trajectory")
    plt.plot(z_detectors[zbegin_index+1:], particle_warmup.xactual[zbegin_index+1:],linestyle="dashed", color="orange")
    plt.plot(curve_z_i, curve_interpolation_warmup, linestyle="dashed", color="orange")
    # plot uncertainty
    plt.errorbar(z_detectors, hits_warmup, yerr=unc_warmup, fmt='.', markersize=3, linewidth=1, color="red", label="Uncertainties Hit positions")
    # plot infos
    plt.xlabel("z [mm]")
    plt.ylabel("x [mm]")
    plt.legend()
    plt.show()


    ### Estimate momentum resolution ###
    # generate 1000 particles
    particles = [particle(p_T_true) for i in range(1000)] 
    print(len(particles))

    # extrapolate trajectories
    for p in particles:
        # calculate change because of magnet
        M_z, M_x = magnet_10_0_5.change(p)

        for z in z_detectors[:zbegin_index+1]:
            cellsHit(p, z) # calculate hits before magnets

        for z in z_detectors[zbegin_index+1:]:
            cellsHit(p, z, zMagnet=magnet_10_0_5.z_end) # calculate hits after magnet


    ## c) generate hitpoints for all 1000 particles and reconstruct p_T
    hitpoints = []
    hitpoints_unc = []
    for p in particles:
        hit , unc = generatePoints(p)
        hitpoints.append(hit)
        hitpoints_unc.append(unc)
    
    p_Ts_reco = []
    p_Ts_unc = []
    x_line_reco_before_magnet = []
    x_line_reco_after_magnet = []
    true_false= []
    for i, p in enumerate(particles):
        # calculate linear regression before magnet
        coeffs_before, cov_before = optimize.curve_fit(line, z_detectors[:zbegin_index+1], hitpoints[i][:zbegin_index+1], sigma=hitpoints_unc[i][:zbegin_index+1], absolute_sigma=True)
        p.s0reco, p.x0reco = coeffs_before
        p.s0recoUncert, p.x0recoUncert = np.sqrt(np.diag(cov_before))

        # calculate linear regression after magnet
        coeffs_after, cov_after = optimize.curve_fit(line, z_detectors[zbegin_index+1:], hitpoints[i][zbegin_index+1:], sigma=hitpoints_unc[i][zbegin_index+1:], absolute_sigma=True)
        p.sMagnetreco, intercept_after_reco = coeffs_after
        p.xMagnetreco = line(magnet_10_0_5.z_end, p.sMagnetreco, intercept_after_reco)
        unc_slope, unc_intercept = np.sqrt(np.diag(cov_after))
        p.sMagnetrecoUncert = unc_slope
        p.xMagnetrecoUncert = np.sqrt((magnet_10_0_5.z_end * unc_slope)**2 + unc_intercept**2)

        # calculate line before magnet
        x_line_reco_before = line(z_detectors[:zbegin_index+1], p.s0reco, p.x0reco)
        x_line_reco_before_magnet.append(x_line_reco_before)

        # calculate line after magnet
        x_line_reco_after = line(z_detectors[zbegin_index+1:], p.sMagnetreco, p.xMagnetreco - p.sMagnetreco * magnet_10_0_5.z_end)
        x_line_reco_after_magnet.append(x_line_reco_after)

        # calculate reconstruction of momentum and the uncertainty
        p_T_reco, p_T_reco_unc = magnet_10_0_5.reconstruct_momentum(p.q, p.s0reco, p.sMagnetreco, p.s0recoUncert, p.sMagnetrecoUncert)
        p_Ts_reco.append(p_T_reco)
        p_Ts_unc.append(p_T_reco_unc)

        true_false.append(np.any(hitpoints_unc[0] == 0))

    print()
    print("z before:", z_detectors[:zbegin_index+1])
    print("z after: ", z_detectors[zbegin_index+1:])
    print()


    ## d) calculate histograms of differences ##
    # prepare plots
    fig, ax = plt.subplots(2, 5, figsize=(20, 12))
    #fig.tight_layout(pad=3.0)
    ax = ax.flatten()

    # calculate differences between reco and true
    p_T_diff = np.array(p_Ts_reco) - p_T_true

    # calculate histogram, mean, std
    hist, bins = np.histogram(p_T_diff, bins=50)
    mean = np.mean(p_T_diff)
    std = np.std(p_T_diff)

    print(f"--- p_T difference histogram statistics p_T_true={p_T_true} GeV, B = {magnet_10_0_5.B} T ---")
    print(f"μ: {mean} [GeV]")
    print(f"σ: {std} [GeV]")
    print()

    # plot histogram
    sns.histplot(p_T_diff, bins="auto", stat="density", kde=True, color="skyblue", ax=ax[0])
    ax[0].axvline(mean, color="red", linestyle="dashed", label=f"μ: {mean:.4f}")
    ax[0].axvline(mean + std, color="orange", linestyle=":", label=f"σ: {std:.4f}")
    ax[0].axvline(mean - std, color="orange", linestyle=":")
    ax[0].set_title("p_T diff histogram, p_T_true=0.3 GeV")
    ax[0].set_xlabel("p_T_reco − p_T_true [GeV]")
    ax[0].set_ylabel("Density")
    ax[0].legend()


    ## e) calculate histogram  of pulls ##
    # prepare plots
    fig2, bx = plt.subplots(2, 5, figsize=(20, 12))
    #fig2.tight_layout(pad=3.0)
    bx = bx.flatten()

    # calculate pull
    p_T_pulls = pull(np.array(p_Ts_reco), p_T_true, p_Ts_unc)

    # calculate histogram, mean, std
    hist, bins = np.histogram(p_T_pulls, bins=50)
    mean = np.mean(p_T_pulls)
    std = np.std(p_T_pulls)

    print(f"--- p_T pull histogram statistics p_T_true={p_T_true} GeV, B = {magnet_10_0_5.B} T ---")
    print(f"μ: {mean} [GeV]")
    print(f"σ: {std} [GeV]")
    print()

    # plot histogram
    sns.histplot(p_T_pulls, bins="auto", stat="density", kde=True, color="skyblue", ax=bx[0])
    bx[0].axvline(mean, color="red", linestyle="dashed", label=f"μ: {mean:.4f}")
    bx[0].axvline(mean + std, color="orange", linestyle=":", label=f"σ: {std:.4f}")
    bx[0].axvline(mean - std, color="orange", linestyle=":")
    bx[0].set_title("p_T pull histogram, p_T_true=0.3 GeV")
    bx[0].set_xlabel("p_T_reco pull")
    bx[0].set_ylabel("Density")
    bx[0].legend()

    # prepare plots for overlay plot
    fig3, cx = plt.subplots(1, 2, figsize=(20, 12))
    cx = cx.flatten()
    fig3.suptitle("Pull distributions overlaid", fontsize=13, fontweight="bold")
    x_ref = np.linspace(-10, 5, 500)
    cx[0].plot(x_ref, norm.pdf(x_ref), linestyle="dashed", color="black", label="N(0,1) ideal")
    cx[1].plot(x_ref, norm.pdf(x_ref), linestyle="dashed", color="black", label="N(0,1) ideal")
    colors_pT = plt.cm.viridis(np.linspace(0, 1, 7))
    colors_B  = plt.cm.plasma(np.linspace(0.2, 1, 4))

    # calculate kde
    cx[0].plot(x_ref, gaussian_kde(p_T_pulls)(x_ref), color=colors_pT[0], label=f"p_T=0.3 GeV (μ={mean:.2f}, σ={std:.2f})")
    cx[1].plot(x_ref, gaussian_kde(p_T_pulls)(x_ref), color=colors_pT[0], label=f"B=0.5 T (μ={mean:.2f}, σ={std:.2f})")


    ### Repeat c)-e) for p_T_true in {0.1,1,2,5,10,20} GeV ###
    ### Estimate momentum resolution ###
    p_T_true_array = [0.1,1,2,5,10,20]

    # generate 1000 particles
    for index, p_T in enumerate(p_T_true_array, 1):
        particles = [particle(p_T) for i in range(1000)] 
        print(len(particles))
        print()

        # extrapolate trajectories
        for p in particles:
            # calculate change because of magnet
            M_z, M_x = magnet_10_0_5.change(p)

            for z in z_detectors[:zbegin_index+1]:
                cellsHit(p, z) # calculate hits before magnets

            for z in z_detectors[zbegin_index+1:]:
                cellsHit(p, z, zMagnet=magnet_10_0_5.z_end) # calculate hits after magnet


        ## generate hitpoints for all 1000 particles and reconstruct p_T
        hitpoints = []
        hitpoints_unc = []
        for p in particles:
            hit , unc = generatePoints(p)
            hitpoints.append(hit)
            hitpoints_unc.append(unc)

        p_Ts_reco = []
        p_Ts_unc = []
        x_line_reco_before_magnet = []
        x_line_reco_after_magnet = []
        true_false= []
        for i, p in enumerate(particles):
            # calculate linear regression before magnet
            coeffs_before, cov_before = optimize.curve_fit(line, z_detectors[:zbegin_index+1], hitpoints[i][:zbegin_index+1], sigma=hitpoints_unc[i][:zbegin_index+1], absolute_sigma=True)
            p.s0reco, p.x0reco = coeffs_before
            p.s0recoUncert, p.x0recoUncert = np.sqrt(np.diag(cov_before))

            # calculate linear regression after magnet
            coeffs_after, cov_after = optimize.curve_fit(line, z_detectors[zbegin_index+1:], hitpoints[i][zbegin_index+1:], sigma=hitpoints_unc[i][zbegin_index+1:], absolute_sigma=True)
            p.sMagnetreco, intercept_after_reco = coeffs_after
            p.xMagnetreco = line(magnet_10_0_5.z_end, p.sMagnetreco, intercept_after_reco)
            unc_slope, unc_intercept = np.sqrt(np.diag(cov_after))
            p.sMagnetrecoUncert = unc_slope
            p.xMagnetrecoUncert = np.sqrt((magnet_10_0_5.z_end * unc_slope)**2 + unc_intercept**2)

            # calculate line before magnet
            x_line_reco_before = line(z_detectors[:zbegin_index+1], p.s0reco, p.x0reco)
            x_line_reco_before_magnet.append(x_line_reco_before)

            # calculate line after magnet
            x_line_reco_after = line(z_detectors[zbegin_index+1:], p.sMagnetreco, p.xMagnetreco - p.sMagnetreco * magnet_10_0_5.z_end)
            x_line_reco_after_magnet.append(x_line_reco_after)

            # when cov matrix has inf values then make it nan
            if np.any(np.isinf(cov_before)) or np.any(np.isinf(cov_after)):
                p_Ts_reco.append(np.nan)
                p_Ts_unc.append(np.nan)
                continue

            # calculate reconstruction of momentum and the uncertainty
            p_T_reco, p_T_reco_unc = magnet_10_0_5.reconstruct_momentum(p.q, p.s0reco, p.sMagnetreco, p.s0recoUncert, p.sMagnetrecoUncert)
            p_Ts_reco.append(p_T_reco)
            p_Ts_unc.append(p_T_reco_unc)

            true_false.append(np.any(hitpoints_unc[0] == 0))
        
        # filter nan/inf values
        valid_mask = np.isfinite(p_Ts_reco) & np.isfinite(p_Ts_unc)

        p_Ts_reco_valid = np.array(p_Ts_reco)[valid_mask]
        p_Ts_unc_valid  = np.array(p_Ts_unc)[valid_mask]

        n_invalid = np.sum(~valid_mask)
        if n_invalid > 0:
            print()
            print(f"{n_invalid} trajectories because of inf/nan removed")
            print()


        ## calculate histograms of differences ##
        # calculate differences between reco and true
        p_T_diff = np.array(p_Ts_reco_valid) - p_T

        # calculate histogram, mean, std
        hist, bins = np.histogram(p_T_diff, bins=50)
        mean = np.mean(p_T_diff)
        std = np.std(p_T_diff)

        print(f"--- p_T difference histogram statistics p_T_true={p_T} GeV, B={magnet_10_0_5.B} T ---")
        print(f"μ: {mean} [GeV]")
        print(f"σ: {std} [GeV]")
        print()

        # plot histogram
        sns.histplot(p_T_diff, bins="auto", stat="density", kde=True, color="skyblue", ax=ax[index])
        ax[index].axvline(mean, color="red", linestyle="dashed", label=f"μ: {mean:.4f}")
        ax[index].axvline(mean + std, color="orange", linestyle=":", label=f"σ: {std:.4f}")
        ax[index].axvline(mean - std, color="orange", linestyle=":")
        ax[index].set_title(f"p_T diff histogram, p_T_true={p_T} GeV")
        ax[index].set_xlabel("p_T_reco − p_T_true [GeV]")
        ax[index].set_ylabel("Density")
        ax[index].legend()


        ## calculate histogram  of pulls ##
        # calculate pull
        p_T_pulls = pull(p_Ts_reco_valid, p_T, p_Ts_unc_valid)

        # calculate histogram, mean, std
        hist, bins = np.histogram(p_T_pulls, bins=50)
        mean = np.mean(p_T_pulls)
        std = np.std(p_T_pulls)

        print(f"--- p_T pull histogram statistics p_T_true={p_T} GeV, B={magnet_10_0_5.B} T ---")
        print(f"μ: {mean} [GeV]")
        print(f"σ: {std} [GeV]")
        print()

        # plot histogram
        sns.histplot(p_T_pulls, bins="auto", stat="density", kde=True, color="skyblue", ax=bx[index])
        bx[index].axvline(mean, color="red", linestyle="dashed", label=f"μ: {mean:.4f}")
        bx[index].axvline(mean + std, color="orange", linestyle=":", label=f"σ: {std:.4f}")
        bx[index].axvline(mean - std, color="orange", linestyle=":")
        bx[index].set_title(f"p_T pull histogram, p_T_true={p_T} GeV")
        bx[index].set_xlabel("p_T_reco pull")
        bx[index].set_ylabel("Density")
        bx[index].legend()

        # plot kde
        cx[0].plot(x_ref, gaussian_kde(p_T_pulls)(x_ref), color=colors_pT[index], label=f"p_T={p_T} GeV (μ={mean:.2f}, σ={std:.2f})")


    ### Repeat c)-e) for B in in {1.0,1.5,2.0} T, p_T_true = 0.3 GeV ###
    ### Estimate momentum resolution ###
    B_array = [1.0,1.5,2.0]

    # generate 1000 particles
    for index, B_i in enumerate(B_array, 7):
        magnet_10 = Magnet(L, B_i, n_before*dZ)
        particles = [particle(p_T_true) for i in range(1000)] 
        print(len(particles))
        print()

        # extrapolate trajectories
        for p in particles:
            # calculate change because of magnet
            M_z, M_x = magnet_10.change(p)

            for z in z_detectors[:zbegin_index+1]:
                cellsHit(p, z) # calculate hits before magnets

            for z in z_detectors[zbegin_index+1:]:
                cellsHit(p, z, zMagnet=magnet_10.z_end) # calculate hits after magnet


        ## generate hitpoints for all 1000 particles and reconstruct p_T
        hitpoints = []
        hitpoints_unc = []
        for p in particles:
            hit , unc = generatePoints(p)
            hitpoints.append(hit)
            hitpoints_unc.append(unc)

        p_Ts_reco = []
        p_Ts_unc = []
        x_line_reco_before_magnet = []
        x_line_reco_after_magnet = []
        true_false= []
        for i, p in enumerate(particles):
            # calculate linear regression before magnet
            coeffs_before, cov_before = optimize.curve_fit(line, z_detectors[:zbegin_index+1], hitpoints[i][:zbegin_index+1], sigma=hitpoints_unc[i][:zbegin_index+1], absolute_sigma=True)
            p.s0reco, p.x0reco = coeffs_before
            p.s0recoUncert, p.x0recoUncert = np.sqrt(np.diag(cov_before))

            # calculate linear regression after magnet
            coeffs_after, cov_after = optimize.curve_fit(line, z_detectors[zbegin_index+1:], hitpoints[i][zbegin_index+1:], sigma=hitpoints_unc[i][zbegin_index+1:], absolute_sigma=True)
            p.sMagnetreco, intercept_after_reco = coeffs_after
            p.xMagnetreco = line(magnet_10.z_end, p.sMagnetreco, intercept_after_reco)
            unc_slope, unc_intercept = np.sqrt(np.diag(cov_after))
            p.sMagnetrecoUncert = unc_slope
            p.xMagnetrecoUncert = np.sqrt((magnet_10.z_end * unc_slope)**2 + unc_intercept**2)

            # calculate line before magnet
            x_line_reco_before = line(z_detectors[:zbegin_index+1], p.s0reco, p.x0reco)
            x_line_reco_before_magnet.append(x_line_reco_before)

            # calculate line after magnet
            x_line_reco_after = line(z_detectors[zbegin_index+1:], p.sMagnetreco, p.xMagnetreco - p.sMagnetreco * magnet_10.z_end)
            x_line_reco_after_magnet.append(x_line_reco_after)

            # when cov matrix has inf values then make it nan
            if np.any(np.isinf(cov_before)) or np.any(np.isinf(cov_after)):
                p_Ts_reco.append(np.nan)
                p_Ts_unc.append(np.nan)
                continue

            # calculate reconstruction of momentum and the uncertainty
            p_T_reco, p_T_reco_unc = magnet_10.reconstruct_momentum(p.q, p.s0reco, p.sMagnetreco, p.s0recoUncert, p.sMagnetrecoUncert)
            p_Ts_reco.append(p_T_reco)
            p_Ts_unc.append(p_T_reco_unc)

            true_false.append(np.any(hitpoints_unc[0] == 0))
        
        # filter nan/inf values
        valid_mask = np.isfinite(p_Ts_reco) & np.isfinite(p_Ts_unc)

        p_Ts_reco_valid = np.array(p_Ts_reco)[valid_mask]
        p_Ts_unc_valid  = np.array(p_Ts_unc)[valid_mask]

        n_invalid = np.sum(~valid_mask)
        if n_invalid > 0:
            print()
            print(f"{n_invalid} trajectories because of inf/nan removed")
            print()


        ## calculate histograms of differences ##
        # calculate differences between reco and true
        p_T_diff = np.array(p_Ts_reco_valid) - p_T_true

        # calculate histogram, mean, std
        hist, bins = np.histogram(p_T_diff, bins=50)
        mean = np.mean(p_T_diff)
        std = np.std(p_T_diff)

        print(f"--- p_T difference histogram statistics p_T_true={p_T_true} GeV, B={magnet_10.B} T ---")
        print(f"μ: {mean} [GeV]")
        print(f"σ: {std} [GeV]")
        print()

        # plot histogram
        sns.histplot(p_T_diff, bins="auto", stat="density", kde=True, color="skyblue", ax=ax[index])
        ax[index].axvline(mean, color="red", linestyle="dashed", label=f"μ: {mean:.4f}")
        ax[index].axvline(mean + std, color="orange", linestyle=":", label=f"σ: {std:.4f}")
        ax[index].axvline(mean - std, color="orange", linestyle=":")
        ax[index].set_title(f"p_T diff histogram, B={magnet_10.B} T")
        ax[index].set_xlabel("p_T_reco − p_T_true [GeV]")
        ax[index].set_ylabel("Density")
        ax[index].legend()


        ## calculate histogram  of pulls ##
        # calculate pull
        p_T_pulls = pull(p_Ts_reco_valid, p_T_true, p_Ts_unc_valid)

        # calculate histogram, mean, std
        hist, bins = np.histogram(p_T_pulls, bins=50)
        mean = np.mean(p_T_pulls)
        std = np.std(p_T_pulls)

        print(f"--- p_T pull histogram statistics p_T_true={p_T_true} GeV, B={magnet_10.B} T ---")
        print(f"μ: {mean} [GeV]")
        print(f"σ: {std} [GeV]")
        print()

        # plot histogram
        sns.histplot(p_T_pulls, bins="auto", stat="density", kde=True, color="skyblue", ax=bx[index])
        bx[index].axvline(mean, color="red", linestyle="dashed", label=f"μ: {mean:.4f}")
        bx[index].axvline(mean + std, color="orange", linestyle=":", label=f"σ: {std:.4f}")
        bx[index].axvline(mean - std, color="orange", linestyle=":")
        bx[index].set_title(f"p_T pull histogram, B={magnet_10.B} T")
        bx[index].set_xlabel("p_T_reco pull")
        bx[index].set_ylabel("Density")
        bx[index].legend()

        # plot kde
        cx[1].plot(x_ref, gaussian_kde(p_T_pulls)(x_ref), color=colors_B[index-7], label=f"B={magnet_10.B} T (μ={mean:.2f}, σ={std:.2f})")

    # plot information cx
    cx[0].axvline(0, color='grey', linewidth=0.8, alpha=0.5)
    cx[0].set_xlim(-10, 5)
    cx[0].set_xlabel("Pull")
    cx[0].set_ylabel("Density")
    cx[0].set_title("Pull vs. p_T_true (B=0.5 T)")
    cx[0].legend(fontsize=7)

    cx[1].axvline(0, color='grey', linewidth=0.8, alpha=0.5)
    cx[1].set_xlim(-10, 5)
    cx[1].set_xlabel("Pull")
    cx[1].set_ylabel("Density")
    cx[1].set_title("Pull vs. B (p_T_true=0.3 GeV)")
    cx[1].legend(fontsize=7)
    fig3.tight_layout()
    plt.show()
    
    
    ### Calculate significance of pull-mean/pull-std difference from 0/1 ###


# run Part 4 Momentum Resolution
Momentum_Resolution()
