import numpy as np
import math

def local_field_d_levels( shape , L, d, W, h_mean = 0):
    wj = np.linspace(-W/2,W/2,num=L)
    dh = 2*wj/(d-1) 
    hn = np.zeros(shape)
    
    if len(shape) == 2:
        for j in range(d):
            hn[j] = wj - dh * j

    elif len(shape) == 3:
        for j in range(d):
            hn[j,j,:] = wj - dh * j

    else:
        print(f'Error : local_field_d_levels not implemented for the current shape {shape}. Return float(nan).')
        return [float('nan')]
    return hn + h_mean



def photon_matter_couplings(alpha):

   

    if len(alpha) == 1:
        theta = alpha[0] * math.pi

        g0  = np.sin(theta)
        g1  = np.cos(theta)
        g   = np.array([g0,g1])

    if len(alpha) == 2:
        theta = alpha[0] * math.pi
        phi   = (0.5-alpha[1]) * math.pi

        g0  = np.sin(theta) * np.sin(phi)
        g1  = np.cos(theta) * np.sin(phi)
        g2  = np.cos(phi)

        g = np.array([g0,g1,g2])

    if len(alpha) > 2:
        print(f'Error : not implemented method for dimension {len(alpha)+1}')
        exit(0)

    return g
