import argparse
import json


def input_mean_field(json_file=''):

    if json_file != '':
        print(f'Loading input from {json}')
        with open(json_file) as j:
            args = json.load(j)
            return args
            
    parser = argparse.ArgumentParser(description='Mean Field calculation of BosonsInCavityState using EM algorithm')

    parser.add_argument("--initial_state", type = str, default = 'coherent_state', help="Initial state")
    parser.add_argument('--L', type = int, default = 2, help = 'Number of sites in the system')
    parser.add_argument('--d', type = int, default = 3, help = 'Number of levels : correspond to SU(d) spins')
    parser.add_argument('--n_photon', type = int, default = 1, help = 'Number of photon modes')
    parser.add_argument('--omega', type = float, default = 1, help = 'frequency of the photon')
    parser.add_argument('--h_mean' , type = float, default = 0, help = 'average linear Zeeman splitting')
    parser.add_argument('--W' , type = float, default = 0, help = 'width linear Zeeman splitting')
    parser.add_argument('--alpha'  , nargs='+' , default = [0.36], help = 'controls photon-matter coupling relations')
    parser.add_argument('--k'      , type = float, default = 0, help = 'photon losses')
    parser.add_argument('--adiabatic_elimination' , type = int, default = 1 , help = '0 : photon is an active DOF ; 1 : photon is enslaved to matter' )
    parser.add_argument('--T'      , type = float, default = 500, help = 'total time' )
    parser.add_argument('--dt_ratio' , type = int, default = 100 , help = 'ratio fastest time scale and dt: fix dt = ratio / max(energy)' )
    parser.add_argument('--number_measures'    , type = int, default = 10000, help = 'number of evenly spaced measures' )

    # optional arguments - are eliminated if not necessary
    parser.add_argument('--theta' , type=float, default=0.4, help = 'angle with z-axis of the magnetic field')
    parser.add_argument('--phi'   , type=float, default=0  , help = 'angle with x-axis of the magnetic field')
    # parser.add_argument('--dtheta', type=float, default=0  , help = 'std of the angle with z-axis of the magnetic field')
    # parser.add_argument('--dphi'  , type=float, default=0  , help = 'std of the angle with x-axis of the magnetic field')

    parser.add_argument("--population", nargs="+", default=[])
    parser.add_argument('--kink', type=int, default = 0 )


    ########################################################################################

    args = vars(parser.parse_args())

    return args

import math


def input_gaussian(json_file=''):

    if json_file != '':
        print(f'Loading input from {json}')
        with open(json_file) as j:
            args = json.load(j)
            return args
            
    parser = argparse.ArgumentParser(description='Mean Field calculation of BosonsInCavityState using EM algorithm')

    parser.add_argument("--initial_state", type = str, default = 'macroscopic_superposition', help="Initial state")
    parser.add_argument('--L', type = int, default = 1, help = 'Number of sites in the system')
    parser.add_argument('--N', type = int, default = 1, help = 'Number of particles within each site')
    parser.add_argument('--N_infty', type = int, default = 1, help = 'N_infty = 1 : N->infty limit ; N_infty = 0 : N = N')
    parser.add_argument('--d', type = int, default = 3, help = 'Number of levels')
    parser.add_argument('--n_photon', type = int, default = 1, help = 'Number of photon modes')
    parser.add_argument('--omega', type = float, default = 1, help = 'frequency of the photon')
    parser.add_argument('--h_mean' , type = float, default = 0, help = 'average linear Zeeman splitting')
    parser.add_argument('--W' , type = float, default = 0, help = 'width linear Zeeman splitting')
    parser.add_argument('--alpha'  , nargs='+' , default = [0.36], help = 'controls photon-matter coupling relations')
    parser.add_argument('--k'      , type = float, default = 0, help = 'photon losses')
    parser.add_argument('--adiabatic_elimination' , type = int, default = 1 , help = '0 : photon is an active DOF ; 1 : photon is enslaved to matter' )
    parser.add_argument('--T'      , type = float, default = 500, help = 'total time' )
    parser.add_argument('--dt_ratio' , type = int, default = 100 , help = 'ratio fastest time scale and dt: fix dt = ratio / max(energy)' )
    parser.add_argument('--number_measures'    , type = int, default = 20000, help = 'number of evenly spaced measures' )

    # optional arguments - are eliminated if not necessary
    # parser.add_argument("--gamma_1", nargs="+", default=[0,math.sqrt(1/3),math.sqrt(2/3)])
    # parser.add_argument("--gamma_2", nargs="+", default=[math.sqrt(2/3),0,math.sqrt(1/3)])

    parser.add_argument("--p", type = float, default=0 , help = 'solely for chaotic dynamics - controls the superposition')

    parser.add_argument("--gamma_1", nargs="+", default=[2/3,1/3,0])
    parser.add_argument("--gamma_2", nargs="+", default=[1/3,0,2/3])
    parser.add_argument('--theta' , type=float, default=0, help = 'relative phase of the superposition of the two bosonic coherent states')
    parser.add_argument('--fraction_quantum_states', type=float, default=1., help = 'fraction of SU(N) states initialized in a cat state. The rest are in a SU(N) coherent state given by gamma_1')

    # parser.add_argument('--kink', type=int, default = 0 )

    ########################################################################################

    args = vars(parser.parse_args())

    return args
