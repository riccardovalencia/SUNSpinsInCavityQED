# Simulate and save number_processor nearly sampled initial conditions
# The initial state is a N-mode Schrodinger cat state specified by parameter so that:
# |psi_cat> ~ |\gamma_1> + |\gamma_2> 
# Goal: compute the Lyapunov exponent as a functin of inhomongeneities in the local fields W and observe that as the 
# the timestep size changes the results do not change dramatically 


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
p_list = np.linspace(0.28,1/3,num=100)
# p_list = []
p_list = np.flip(p_list)

x_list = []

for p in p_list:
    x_list.append([f'{1/3+p:.12f}', f'{1/3:.12f}', f'{1/3 - p:.12f}'])

lyapunov_exponent  = []
dlyapunov_exponent = []


args = input.input_gaussian(json_file)


dt_ratio_list = [500]

for dt_ratio in dt_ratio_list:
    main_dir = f"/home/ricval/Documenti/Cavity_python/data_paper/SU3_Lyapunov_W{args['W']:.2f}_1E-8_dt{dt_ratio}"

    if not os.path.isdir(main_dir):
        Path(main_dir).mkdir()

    args['dt_ratio'] = dt_ratio

    for idx, x in enumerate(x_list):

        O = {}
        for i in range(args['d']):
            for j in range(i+1,args['d']):
                O[f's{i}{j}'] = []

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
                folder = f"{main_dir}/SU{args['d']}_Gaussian_L{args['L']}_alpha{args['alpha'][0]}_p{p_list[idx]:.3f}_W{args['W']:.3f}_index{index_r}"

                if not os.path.isdir(folder):
                    Path(folder).mkdir()

                for name in list(O.keys()):
                    np.save(f"{folder}/{name}.npy",np.array(obs[name]))
                np.save(f"{folder}/timesteps.npy",timesteps)         
            
                