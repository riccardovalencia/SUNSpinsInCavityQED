# mean field evolution of a d level bosonic system
# each mode is in a bosonic coherent state
import sys
sys.path.append(f'../library')
import beta_functions, data_management, state, evolution, parameters, mean_field_initial_states, input
import math
import numpy as np
import matplotlib.pyplot as plt

main_dir = "Fig_2/"
do_plots = True
show_plots = True
json_file = ""

if __name__ == "__main__":

    # input and some operations
    args = input.input_mean_field(json_file)
    args_json = args.copy()

    alpha = np.array([float(y) for y in args['alpha']])        
    for k , a in enumerate(alpha):
        args[f'alpha{k}'] = alpha[k]
    args.pop('alpha')

    g = parameters.photon_matter_couplings(alpha)
    g *= np.sqrt(np.abs(args['omega']))
    
    # initial state
    psi = state.State_bosonic_coherent_state(args['L'],args['d'])
    success = mean_field_initial_states.initialize_state(psi,args)

    if success == -1:
        print('Something is wrong in the initialization...')
        exit(0)

    # parameters 

    hn = parameters.local_field_d_levels( psi.matter_shape , args['L'], args['d'], args['W'])
    dt = 1/(args['dt_ratio']*np.abs(args['omega']))
    args['number_measures'] = min(args['number_measures'],int(args['T']/dt))

    
    # create folder and csv file for storing and keepin track data

    folder , success = data_management.create_directory_and_csv(main_dir,f"mean_field_SU{psi.d}_{args['initial_state']}",list(args.keys()))
    exist = data_management.check_existence_data(folder,args)
    index = data_management.get_index_csv(folder,args)

    if exist == 1:
        print('Directory exist')
        exit(0)

    elif exist == -1:
        print('Directory does not exist, but header do not match the key')
        exit(0)

    elif exist == 0:
        print('Directory does not exist and headers match. Start the simulation.')

    # changing args for performing time evolution

    args_csv = args.copy()

    args['hn'] = hn
    args['g']  = g
    args['dt'] = dt

    # time evolution
    
    obs , times = evolution.perform_evolution_and_measure(psi,beta_functions.beta_function_BiC_SUN_mean_field_RWA,args)

    obs['stot'] = 0

    for i in range(args['d']-1):
        obs['stot'] += g[i] * np.array(obs[f's{i}{i+1}'])
    # index = data_management.get_index_csv(folder,args)
    # index  = data_management.get_last_index(folder,args)
    folder , _ = data_management.save_data(folder,'data','mean_field_SU',index,args['d'],args_csv,args_json, obs,times)


    # plots

    if do_plots:
        fig , ax           = plt.subplots(2,math.comb(psi.d,2),figsize=(16,10))
        fig_zoom , ax_zoom = plt.subplots(2,math.comb(psi.d,2),figsize=(16,10))
        ax = ax.flatten()
        ax_zoom = ax_zoom.flatten()
        index_zoom = int(len(times)*0.85)

        for idx, O in enumerate(obs):
            ax[idx].plot(times,np.abs(np.array(obs[O])))
            ax[idx].set_title(f'{O}')
            ax_zoom[idx].plot(times[index_zoom:],np.abs(np.array(obs[O][index_zoom:])))
            ax_zoom[idx].set_title(f'{O}')
            

        plt.tight_layout()

        fig.savefig(f'{folder}/obs.png')
        fig_zoom.savefig(f'{folder}/obs_zoom.png')

        if show_plots:
            plt.show()

