# Simulate and save number_processor nearly sampled initial conditions
# The initial state is a N-mode Schrodinger cat state specified by parameter so that:
# |psi_cat> ~ |\gamma_1> + |\gamma_2> 
# Goal: compute the Lyapunov exponent as a functin of inhomongeneities in the local fields W

# NB: it is used the function compute_lyapunov_exponent and not compute_average_lyapunov_exponent
# the one used is DEPRECATED.

import sys
import concurrent.futures 
import multiprocessing

sys.path.append(f'./library')
import input, state, parameters, mean_field_initial_states, data_management, evolution, beta_functions, chaos
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from pathlib import Path

number_processors = multiprocessing.cpu_count()
# number_processors = 1
main_dir = 'lyapunov_gaussian_states_swipe_W/'
json_file = ""
save_data = False
name_swipe = "W"
# x_list = [500,1000,2000]
# x_list = [2000,4000]
x_list = [0.05,0.1,0.8,1.2,1.5]
x_list = [0,0.01]

x_plot = x_list
x_list_plot = x_list

lyapunov_exponent = []
dlyapunov_exponent = []
if len(x_plot) > 0:
    fig, ax = plt.subplots(1,len(x_plot))
    if len(x_plot) == 1:
        ax = [ax]

fig_lyap, ax_lyap = plt.subplots(1,) 

args = input.input_gaussian(json_file)

for idx, x in enumerate(x_list):

    O = {}
    for i in range(args['d']):
        for j in range(i+1,args['d']):
            O[f's{i}{j}'] = []
    # O['a'] = []

    args[f'{name_swipe}'] = x


    with concurrent.futures.ProcessPoolExecutor() as executor:

        results = [executor.submit(evolution.SUN_gaussian_dynamics, args,index,True) for index in range(number_processors)]
        for index, f in enumerate(results):
            result = f.result()
            obs        = result[1]
            timesteps  = result[2]

            for name in list(O.keys()):
                O[name].append(np.abs(np.array(obs[name])))

            
            if x in x_plot:

                ax[x_plot.index(x)].plot(timesteps,O['s01'][-1],linestyle='-',linewidth=0.7)
                ax[x_plot.index(x)].set_title(f'{x}')
                # ax[0].plot(timesteps,O['s01'][-1],linestyle='-',linewidth=0.7)
                # ax[0].set_title(f'{x}')

            folder = f"{main_dir}SUN_Gaussian_L{args['L']}_W{args['W']:.3f}_index{index}"
            if not os.path.isdir(folder):
                Path(folder).mkdir()

            for name in list(O.keys()):
                np.save(f"{folder}/{name}.npy",np.abs(np.array(obs[name])))
            np.save(f"{folder}/timesteps.npy",timesteps)    


    l , t,  dO = chaos.compute_lyapunov_exponent(O,timesteps)
    lyapunov_exponent.append(l[0])
    dlyapunov_exponent.append(l[1])



ax_lyap.plot(x_list_plot,lyapunov_exponent,linestyle='-',color='black')
ax_lyap.errorbar(x_list_plot,lyapunov_exponent,yerr=dlyapunov_exponent,linestyle='-',color='black')
plt.show()