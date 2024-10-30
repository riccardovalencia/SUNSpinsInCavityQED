# Optimization procedure in order to fit via a single-body effective Hamiltonian
# the many-body dynamics of the observables of interest.
# The optimization procedure is based on minimized the norm-1 distance between the
# effective one-body reduced density matrix and the average one-body reduced density matrix
# obtained via the many-body dynamics.
# The procedure works well for \theta=0.48\pi and W=0.1 (so that g_+ = \cos(\theta) and g_- = \sin(\theta))



import sys
from time import time
sys.path.append(f'./library')
import beta_functions, data_management, state, evolution, parameters, gaussian_initial_states, input
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import json
from pathlib import Path

mean_field = True
gaussian  = False

def final_trajectory(x,args):

    psi = state.State_bosonic_gaussian_state(args['L'],args['d'])
    C_t0 = np.zeros(psi.matter_shape[:2],dtype=complex)

    index = index_g
    hn = parameters.local_field_d_levels( psi.matter_shape , args['L'], args['d'], args['W'])
    hn[:,:,0] = np.diag(x[:3],k=0)
    args['hn'] = hn

    for n in range(args['d']):
        C_t0[n,n] += x[index]
        index += 1


    for n in range(args['d']):
        for k in range(n+1,args['d']):
            C_t0[n,k] += x[index]
            index += 1

    
    for n in range(args['d']):
        for k in range(n+1,args['d']):
            C_t0[n,k] += 1j*x[index]
            index += 1
    
    
    C_t0 += np.transpose(np.conj(np.triu(C_t0,k=1)))

    psi.update_single_site(C_t0,0)

    g_matrix = np.zeros(psi.matter_shape,dtype=np.dtype(np.complex128))   
    for j in range(args['L']):
        g_matrix[:,:,j] += np.diag(x[3:index_g],k=1)

    args['g'] = g_matrix
    
    obs_G , time = evolution.perform_evolution_and_measure(psi,beta_functions.beta_function_BiC_SUN_gaussian_RWA,args)

    
    return obs_G, time




def distance_function(x,t,args,x_target):

    # print(args)

    # exit(0)
    psi = state.State_bosonic_gaussian_state(args['L'],args['d'])
    C_t0 = np.zeros(psi.matter_shape[:2],dtype=complex)

    index = index_g
    hn = parameters.local_field_d_levels( psi.matter_shape , args['L'], args['d'], args['W'])
    hn[:,:,0] = np.diag(x[:3],k=0)
    args['hn'] = hn
    # print(x)
    # # exit(0)

    for n in range(args['d']):
        C_t0[n,n] += x[index]
        index += 1


    for n in range(args['d']):
        for k in range(n+1,args['d']):
            C_t0[n,k] += x[index]
            index += 1

    
    for n in range(args['d']):
        for k in range(n+1,args['d']):
            C_t0[n,k] += 1j*x[index]
            index += 1
    
    
    C_t0 += np.transpose(np.conj(np.triu(C_t0,k=1)))

    psi.update_single_site(C_t0,0)


    g_matrix = np.zeros(psi.matter_shape,dtype=np.dtype(np.complex128))   
    for j in range(args['L']):
        g_matrix[:,:,j] += np.diag(x[3:index_g],k=1)

    args['g'] = g_matrix
    # print(args['g'])

    # print(g_matrix[:,:,0])
    # print(C_t0)
    # print(args)
    # exit(0)
    obs_G , _ = evolution.perform_evolution_and_measure(psi,beta_functions.beta_function_BiC_SUN_gaussian_RWA,args)

    if 'a' in obs_G:
        obs_G.pop('a')

    distance = 0
    for name in list(obs_G.keys()):
        distance += np.average(np.abs(x_target[name]-np.array(obs_G[name]))**2)
    
    print(distance)
    return distance





main_dir = ''
json_file = ""
save_data = False
name_swipe = "W"

fig, ax = plt.subplots(1,1)



if mean_field:

    args = input.input_mean_field(json_file)
    psi_L, obs, times = evolution.SUN_mean_field_dynamics(args,0,noise=False,measure_population=True)  

elif gaussian:
    args = input.input_gaussian(json_file)
    psi_L, obs, times = evolution.SUN_gaussian_dynamics(args,0,noise=False,measure_population=True)  

if 'a' in obs:
    obs.pop('a')
print(list(obs.keys()))

args['L'] = 1
psi = state.State_bosonic_gaussian_state(args['L'],args['d'])
C_t0 = np.zeros(psi.matter_shape,dtype=complex)

start = int(len(times)/2)
# start = 0
args['T'] -= times[start]

for i in range(args['d']):
    for j in range(args['d']):
        if i > j:
            C_t0[i,j,0] = np.conj(obs[f's{j}{i}'][start])
        else:
            C_t0[i,j,0] = obs[f's{i}{j}'][start]

print(C_t0[:,:,0])
# x0[0] = hn
x0 = [0,0,0]


alpha = np.array([float(y) for y in args['alpha']])        
g  = parameters.photon_matter_couplings(alpha)
g *= np.sqrt(args['omega'])

for item in g:
    x0 += [item]

index_g = len(x0)
for n in range(args['d']):
    x0 += [C_t0[n,n,0]]

for n in range(args['d']):
    for k in range(n+1,args['d']):
        x0 += [np.real(C_t0[n,k,0])]

for n in range(args['d']):
    for k in range(n+1,args['d']):
        x0 += [np.imag(C_t0[n,k,0])]

x_target = {}

for name in list(obs.keys()):
    x_target[name] = obs[name][start:]

# print(len(x0))
# exit(0)
dt = 1/(args['dt_ratio']*args['omega'])
# print(args['number_measures'])
args['number_measures'] = int(len(times[start:]))
args['dt'] = dt

print(args)



res = scipy.optimize.minimize(
    distance_function,
    method='Nelder-Mead',
    x0 = x0,
    args =(times,args,x_target),
    tol=1e-3,
    options={'maxiter':100,'adaptive':True}
)
args['number_measures'] = int(len(times))

obs_G, times_G = final_trajectory(res.x,args)
if 'a' in obs_G:
    obs_G.pop('a')

fig, ax = plt.subplots(1,len(obs.keys()))

for idx, name in enumerate(obs_G):
    ax[idx].plot(times,np.abs(np.array(obs[name])))
    ax[idx].plot(times_G+times[start],np.abs(np.array(obs_G[name])),linestyle='--',color='black')

save_folder = f"SU{args['d']}_W{args['W']:.2f}_reductions"

if os.path.isdir(save_folder)==False:
    Path(save_folder).mkdir()



# with open(f'{save_folder}/input.json','w') as f:
#     json.dump(args,f)

for _ , O in enumerate(obs_G):
    np.save(f'{save_folder}/{O}.npy',obs_G[O])
np.save(f'{save_folder}/time.npy',times_G+times[start])


plt.show()