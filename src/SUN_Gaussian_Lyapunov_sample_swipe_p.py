# Simulate and save number_processor nearly sampled initial conditions
# The initial state is a N-mode Schrodinger cat state specified by parameter so that:
# |psi_cat> ~ |\gamma_1> + |\gamma_2> 
# Goal: sample trajectories so that AFTER we compute the Lyapunov exponent as a function of  of fraction Q of initial SU(N) spins in a Gaussian state


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
p_list = np.linspace(0.29,1/3,num=20)
p_list = np.flip(p_list)
p_list = [0.299,0]
# p_list = np.linspace(0,0.1,num=100)
# p_list = [0.3005, 0.3118,0.3123, 0.3306, 0.3312]
x_list = []
# p_list = [0.26,0.27,0.28,0.29]
# p_list = [0.29 + 0.0005 * j for j in range(20)]
# p_list = np.
# p_list = [0 , 0.05, 0.10 ,0.15 ,0.20 ,0.25, 0.30]
# p_list += [0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.333]
# p_list += [0.305, 0.310, 0.315, 0.320, 0.330]

for p in p_list:
    x_list.append([f'{1/3+p:.12f}', f'{1/3:.12f}', f'{1/3 - p:.12f}'])
    # x_list.append([f'{math.sqrt(1/3+p):.12f}', f'{math.sqrt(1/3):.12f}', f'{math.sqrt(1/3 - p):.12f}'])

lyapunov_exponent  = []
dlyapunov_exponent = []


args = input.input_gaussian(json_file)

# print(args['dt_ratio'])
# exit(0)

main_dir = f"SU{args['d']}_Lyapunov_swipe_p_W{args['W']:.2f}"
main_dir = '/home/ricval/Documenti/Cavity_python/data_paper/SU3_Gaussian_p_vs_W'
# main_dir = '/home/ricval/Documenti/Cavity_python/data_paper/SU3_Lyapunov_swipe_p_W0.00_1E-8'
# main_dir = '/home/ricval/Documenti/Cavity_python/data_paper/SU3_Lyapunov_W0_1E-8'

if not os.path.isdir(main_dir):
    Path(main_dir).mkdir()


for idx, x in enumerate(x_list):

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

            folder = f"{main_dir}/SUN_Gaussian_L{args['L']}_p{p_list[idx]:.3f}_W{args['W']:.3f}_index{index_r}"
            # folder = f"{main_dir}/SU{args['d']}_Gaussian_L{args['L']}_alpha{args['alpha'][0]}_p{p_list[idx]:.3f}_W{args['W']:.3f}_index{index_r}"

            if not os.path.isdir(folder):
                Path(folder).mkdir()

            for name in list(O.keys()):
                np.save(f"{folder}/{name}.npy",np.array(obs[name]))
            np.save(f"{folder}/timesteps.npy",timesteps)         
          
            