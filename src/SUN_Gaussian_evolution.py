from json.tool import main
import sys
import concurrent.futures 
import multiprocessing

sys.path.append(f'./library')
import input, state, parameters, mean_field_initial_states, data_management, evolution, beta_functions, chaos
import numpy as np
import math
from pathlib import Path
import os

number_processors = multiprocessing.cpu_count()
# number_processors = 1
main_dir = 'SU3_Gaussian_swipe_p_and_W/'
main_dir = ''
json_file = ""
save_data = False
name_swipe = "L"
# x_list = [500,1000,2000]
x_list = [1]


args = input.input_gaussian(json_file)

x = [f"{np.sqrt(1/3+args['p']):.12f}", f'{np.sqrt(1/3):.12f}', f"{np.sqrt(1/3 - args['p']):.12f}"]

args['gamma_1'] = x
args['gamma_2'] = np.roll(x,-1)

for _ in range(1):
    O = {}
    for i in range(args['d']):
        for j in range(i+1,args['d']):
            O[f's{i}{j}'] = []

    #args[f'{name_swipe}'] = x

    with concurrent.futures.ProcessPoolExecutor() as executor:

        results = [executor.submit(evolution.SUN_gaussian_dynamics, args,index,True) for index in range(number_processors)]
        for index, f in enumerate(results):
            result = f.result()
            obs        = result[1]
            timesteps  = result[2]

            for name in list(O.keys()):
                O[name].append(np.array(obs[name]))

            folder = f"{main_dir}SU{args['d']}_Gaussian_{args['initial_state']}_L{args['L']}_p{args['p']:.3f}_W{args['W']:.3f}_index{index}_TEST"
            if not os.path.isdir(folder):
                Path(folder).mkdir()

            for name in list(O.keys()):
                np.save(f"{folder}/{name}.npy",np.array(obs[name]))
            np.save(f"{folder}/timesteps.npy",timesteps)    
            
                    
            

#     # l , t,  dO = chaos.compute_lyapunov_exponent(O,timesteps)
#     # lyapunov_exponent.append(l)



# # ax_lyap.plot(x_list_plot,lyapunov_exponent,linestyle='-',color='black')
# plt.show()
