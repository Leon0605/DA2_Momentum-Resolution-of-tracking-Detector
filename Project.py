import numpy as np
import math

class particle:

    def __init__(self):
        self.x0 = 0
        self.z0 = np.random.normal(0,10,None)
        self.s0 = np.random.normal(0,0.1,None)
        self.layers = None

def cellsHit(dZ,n,z):



