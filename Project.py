import numpy as np
import math
from sklearn.linear_model import LinearRegression

class particle:

    def __init__(self):
        self.z0 = 0
        self.x0 = np.random.normal(0,1,None)
        self.s0 = np.random.normal(0,0.1,None)
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
    p.layers.append(math.floor(x/cellWidth)+1) #calculate and store cell index

def generatePoints(p):
    genX = []
    for i in range(0,n):
        genX.append(np.random.uniform(p.layers[i]-1*cellWidth, (p.layers[i])*cellWidth)) #generate data from hit cell index by drawing from cell interval as uniform distribution
    return np.array(genX)



def run():
    p1 = particle() #initialize a single particle
    Z = []
    print(p1.s0)
    for i in range(0, n):
        z = (i+1)*dZ
        Z.append(z)
        cellsHit(p1, z)
    Z = np.array(Z)
    X = generatePoints(p1)
    print(X)
    lin = LinearRegression().fit(Z.reshape(-1,1),X)
    print(lin.coef_[0])
    print(p1.layers)

run()