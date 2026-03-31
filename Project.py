import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import optimize
from sklearn.linear_model import LinearRegression

class particle:

    def __init__(self):
        self.z0 = 0
        self.x0 = np.random.normal(0,1,None)
        self.x0reco = 0
        self.x0recoUncert = 0
        self.s0 = np.tan(np.random.normal(0,0.1,None))
        self.s0reco = 0
        self.s0recoUncert = 0
        self.t0 = 0
        self.xactual = []
        self.layers = []

#Setup Parameters
n=5 #number of layers
dZ = 20 #distance between layers 2cm = 20mm
cellWidth = 0.5 #500 micrometer = 0.5mm

def cellsHit(p, z):
    x = p.x0 + p.s0 * (z) #calculate x position
    p.xactual.append(x)
    if(x >= 0):
        p.layers.append(math.floor(x/cellWidth)+1)
    else:
        p.layers.append(math.floor(x/cellWidth)) #calculate and store cell index

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
    for i in range(0, n):
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
    z_line = np.linspace(0, n*dZ, 10)
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
simulation()
