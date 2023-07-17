#import python modules
import numpy as np
import sys
import os

def initialize_positions(N, L):
    """input: total number of atoms (N), box size (L)
    output: positions [(N,3) array]
"""
    #start with zeros
    pos = np.zeros((N,3), float)
    
    #create integer grid locations for cubic lattice
    LocLat = int(N**(1./3.) + 1.)
    #print(LocLat)
    SpacLat = L / LocLat
    #print(SpacLat)    
    
    #create lattice sites
    Lat = SpacLat * np.arange(LocLat, dtype=float) - 0.5*L
    #print(Lat)
    
    #go over x, y, z points
    i = 0
    for x in Lat:
        for y in Lat:
            for z in Lat:
                
                pos[i] = np.array([x,y,z], float)
                
                #random diplacement to help initial CG minimization
                RandDisp = 0.1 * SpacLat * (np.random.rand(3) - 0.5)
                pos[i] = pos[i] + RandDisp
                
                i += 1
                if i >= N:
                    return pos
    return pos


def velocity_rescaler(v, T):
    """input: velocities [(N,3) array] and target temperature (T)
    output: rescaled velocities [(N,3) array]
"""
    # first apply condition for zero net momentum
    v = v - v.mean(axis=0)
    
    #compte the kinetic energy
    kin_energy = 0.5 * np.sum(v * v)
    
    #compute lambda, the rescaling factor 
    lam = np.sqrt(1.5 * len(v) * T / kin_energy)
    v = v * lam
    
    return v


def initialize_velocities(N, T):
    """input: total number of atoms (N) and target tempearture (T)
    output: initial velocities [(N,3) array]
"""
    v = np.random.rand(N, 3)
    v = velocity_rescaler(v, T)

    return v

def calc_temperature(v):
    """input: velocities [(N,3) array] 
    output: instantaneous temperature (T)
"""
    T = np.sum(v * v) / (3. * len(v))
    return T

