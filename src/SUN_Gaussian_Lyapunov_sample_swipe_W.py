# Simulate and save number_processor nearly sampled initial conditions
# The initial state is a N-mode Schrodinger cat state specified by parameter so that:
# |psi_cat> ~ |\gamma_1> + |\gamma_2> 
# Goal: sample trajectories so that AFTER we compute the Lyapunov exponent as a functin of inhomongeneities in the local fields W

from json.tool import main
import sys
import concurrent.futures 
import multiprocessing

sys.path.append(f'./library')
import input, state, parameters, mean_field_initial_states, data_management, evolution, beta_functions, chaos
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
import os

number_processors = multiprocessing.cpu_count()
# number_processors = 1
main_dir = 'lyapunov_gaussian_states_swipe_W/'
json_file = ""
save_data = False
name_swipe = "W"
# x_list = [500,1000,2000]
x_list = [0]


args = input.input_gaussian(json_file)
args['gamma_2'] = np.roll(args['gamma_1'],-1)

for idx, x in enumerate(x_list):

    O = {}
    for i in range(args['d']):
        for j in range(i+1,args['d']):
            O[f's{i}{j}'] = []

    args[f'{name_swipe}'] = x

    with concurrent.futures.ProcessPoolExecutor() as executor:

        results = [executor.submit(evolution.SUN_gaussian_dynamics, args,index,True) for index in range(number_processors)]
        for index, f in enumerate(results):
            result = f.result()
            obs        = result[1]
            timesteps  = result[2]

            for name in list(O.keys()):
                O[name].append(np.abs(np.array(obs[name])))

            folder = f"{main_dir}SUN_Gaussian_L{args['L']}_W{args['W']:.3f}_index{index}"
            if not os.path.isdir(folder):
                Path(folder).mkdir()

            for name in list(O.keys()):
                np.save(f"{folder}/{name}.npy",np.abs(np.array(obs[name])))
            np.save(f"{folder}/timesteps.npy",timesteps)            
            

#     # l , t,  dO = chaos.compute_lyapunov_exponent(O,timesteps)
#     # lyapunov_exponent.append(l)



# # ax_lyap.plot(x_list_plot,lyapunov_exponent,linestyle='-',color='black')
# plt.show()