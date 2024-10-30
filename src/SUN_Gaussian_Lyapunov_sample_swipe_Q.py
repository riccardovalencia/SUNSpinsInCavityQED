# Simulate and save number_processor nearly sampled initial conditions
# Goal:  sample trajectories so that AFTER we compute the Lyapunov exponent as a functin of fraction Q of initial SU(N) spins in a Gaussian state
# The Gaussian state is a N-mode Schrodinger cat state specified by parameter so that:
# |psi_cat> ~ |\gamma_1> + |\gamma_2> 
# The remaining fraction 1-Q of sites is initialized in |\gamma_1> (coherent state)

import sys
import concurrent.futures 
import multiprocessing

sys.path.append(f'./library')
import input, evolution, chaos
import numpy as np
from scipy.signal import argrelextrema
import os, math
from pathlib import Path


t_transient = 30
number_processors = multiprocessing.cpu_count()
# number_processors = 1
main_dir = ''
json_file = ""
save_data = False


name_swipe = "gamma"

p_list = np.linspace(0.3,1/3,num=20)
p_list = [0.30,0.305,0.310,0.315,0.320,0.325,1/3]
p_list = list([0.30 + 0.002 * j for j in range(1,3)])
p_list += list([0.305 + 0.002 * j for j in range(1,3)])
p_list += list([0.310 + 0.002 * j for j in range(1,3)])
p_list += list([0.315 + 0.002 * j for j in range(1,3)])
p_list += list([0.320 + 0.002 * j for j in range(1,3)])
p_list += list([0.325 + 0.002 * j for j in range(1,3)])
p_list += [0.331, 0.332]

p_list = [0.33]
Q_list = np.linspace(0,1,num=20)
Q_list = list([j * 0.05 for j in range(21)])
Q_list = np.array(Q_list)
Q_list = Q_list[Q_list>0.35]

# print(len(p_list)*len(Q_list))
# exit(0)
x_list = []


for p in p_list:
    x_list.append([f'{1/3+p:.12f}', f'{1/3:.12f}', f'{1/3 - p:.12f}'])


args = input.input_gaussian(json_file)
args['initial_state'] = 'quantum_bubble'

main_dir = '/home/ricval/Documenti/Cavity_python/data_paper/SU3_Lyapunov_quantumbubble_p_Q_W0.00_1E-8'

if not os.path.isdir(main_dir):
    Path(main_dir).mkdir()

args['L'] = 1000
for idx, x in enumerate(x_list):

    for idx_Q, Q in enumerate(Q_list):

        print(f"p={p_list[idx]} Q={Q}")


        args['fraction_quantum_states'] = Q
        O = {}
        for i in range(args['d']):
            for j in range(i+1,args['d']):
                O[f's{i}{j}'] = []

        # args[f'{name_swipe}'] = x
        args['gamma_1'] = x
        args['gamma_2'] = np.roll(x,-1)

        
        with concurrent.futures.ProcessPoolExecutor() as executor:

            results = [executor.submit(evolution.SUN_gaussian_dynamics, args,index) for index in range(number_processors)]
            for index_r, f in enumerate(results):
                result = f.result()
                obs        = result[1]
                timesteps  = result[2]
                
                transient = (timesteps > t_transient)[0]


                for name in list(O.keys()):
                    # O[name].append(np.abs(np.array(obs[name])))
                    O[name].append(np.array(obs[name]))

                # folder = f"{main_dir}/SU{args['d']}_Gaussian_L{args['L']}_W{args['W']:.3f}_p{p_list[idx]:.4f}_index{index_r}"
                folder = f"{main_dir}/SU{args['d']}_Gaussian_L{args['L']}_alpha{args['alpha'][0]}_Q{Q:.3f}_p{p_list[idx]:.3f}_W{args['W']:.3f}_index{index_r}"

                if not os.path.isdir(folder):
                    Path(folder).mkdir()

                for name in list(O.keys()):
                    np.save(f"{folder}/{name}.npy",np.array(obs[name]))
                np.save(f"{folder}/timesteps.npy",timesteps)         
          
            