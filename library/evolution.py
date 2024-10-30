from scipy.integrate import ode
import numpy as np
import beta_functions, data_management, parameters, mean_field_initial_states, gaussian_initial_states, state
import sys

class Evolution():

    def __init__(self,initial_state,beta_function,args,method='zvode',rtol=1E-10):
        self.integrator_solver = ode(beta_function).set_integrator(method, method='bdf',rtol=rtol)
        self.integrator_solver.set_f_params([args,[initial_state.matter_shape,initial_state.photon_shape]])
        self.integrator_solver.set_initial_value(initial_state.flatten())
        self.dt  = args["dt"]
        self.tau = args["dt"]
        self.state = initial_state.flatten()

    def evolve(self,T):

        while self.tau <= T:
            self.state = self.integrator_solver.integrate(self.tau)
            self.tau += self.dt
        
            # sol = self.integrator_solver.integrate(self.tau)
            # print(self.state)
            # print(sol)


            

class Evolution_real_imag():

    def __init__(self,initial_state,beta_function,args,method='zvode',rtol=1E-10):
        self.integrator_solver = ode(beta_function).set_integrator(method, method='bdf',rtol=rtol)
        self.integrator_solver.set_f_params([args,[initial_state.matter_shape,initial_state.photon_shape]])
        self.integrator_solver.set_initial_value(initial_state.flatten_real_imag())
        self.dt  = args["dt"]
        self.tau = args["dt"]
        self.state = initial_state.flatten_real_imag()

    def evolve(self,T):

        while self.tau <= T:
            self.state = self.integrator_solver.integrate(self.tau)
            # sol = self.integrator_solver.integrate(self.tau)
            # print(self.state)
            # print(sol)
            self.tau += self.dt


            


def perform_evolution_and_measure(psi,beta_function,args,method='zvode',measure_population=False,measure_abs=False,measure_connected=0,measure_purity=1,rtol=1E-10):

    times = np.linspace(args['dt'],args['T'],num=args['number_measures'])

    obs = {}
    for i in range(psi.d):
        for j in range(i+1,psi.d):
            obs[f's{i}{j}'] = []
    obs['a'] = []

    if measure_population:
        for i in range(psi.d):
            obs[f's{i}{i}'] = []

    if measure_connected == 1:
        for i in range(psi.d):
            for j in range(i+1,psi.d):
                obs[f's{i}{j}_c'] = []

    if measure_purity == 1:
        obs['purity_av'] = []

    evol = Evolution(psi,beta_function,args,method,rtol)

    print('Evolving...')

    for idx, t in enumerate(times):
        evol.evolve(t)
        psi.update(evol.state)

        

        

        if t in times[::int(len(times)/20)]:
            print(f'{idx/len(times)*100:.2f}%')
            # print(f'Size of class evol : {sys.getsizeof(evol.state)}')
            # size =  sys.getsizeof(obs['s01'])  #* 1000000 / 10**6
            # print(size)
            # for name in list(obs.keys()):
            #     print(f'Size of {name} : {sys.getsizeof(obs[name])/ 10**6}')
            
        for i in range(psi.d):
            for j in range(i+1,psi.d):
                if measure_abs:
                    obs[f's{i}{j}'] += [np.abs(psi.measure_coherence_av(i,j))]                
                else:
                    obs[f's{i}{j}'] += [psi.measure_coherence_av(i,j)]
        
        if measure_population:
            for i in range(psi.d):
                obs[f's{i}{i}'] += [np.abs(psi.measure_coherence_av(i,i))]

        if measure_connected == 1:
            for i in range(psi.d):
                for j in range(i+1,psi.d):
                    if measure_abs:
                        obs[f's{i}{j}_c'] += [np.abs(psi.measure_connected_coherence_av(i,j))]                
                    else:
                        obs[f's{i}{j}_c'] += [psi.measure_connected_coherence_av(i,j)]

        if measure_purity == 1:
            obs['purity_av'] += [psi.measure_purity()]
        
        obs['a'] += [psi.measure_photon_amplitude()]

        # for name in list(obs.keys()):
        #     print(f'Size of {name} : {sys.getsizeof(obs[name])}')
            
    
    return obs, times




def perform_evolution_and_measure_real_imag(psi,beta_function,args,method='zvode',rtol=1E-10):


    times = np.linspace(args['dt'],args['T'],num=args['number_measures'])

    obs = {}
    for i in range(psi.d):
        for j in range(i+1,psi.d):
            obs[f's{i}{j}'] = []
    obs['a'] = []
    print(method)
    evol = Evolution_real_imag(psi,beta_function,args,method,rtol)

    print('Evolving...')

    for idx, t in enumerate(times):
        evol.evolve(t)
        psi.update_real_imag(evol.state)

        if t in times[::int(len(times)/20)]:
            print(f'{idx/len(times)*100:.2f}%')
            
        for i in range(psi.d):
            for j in range(i+1,psi.d):
                obs[f's{i}{j}'] += [psi.measure_coherence_av(i,j)]
        
        obs['a'] += [psi.measure_photon_amplitude()]
    

    return obs, times





def SUN_mean_field_dynamics(args,index,noise=True,save_data=False,main_dir='',measure_population=False):

    args_json = args.copy()
    args_single = args.copy()
    

    alpha = np.array([float(y) for y in args['alpha']])        
    for k , _ in enumerate(alpha):
        args_single[f'alpha{k}'] = alpha[k]
    args_single.pop('alpha')

    g  = parameters.photon_matter_couplings(alpha)
    g *= np.sqrt(args_single['omega'])

    # initial state

    psi = state.State_bosonic_coherent_state(args_single['L'],args_single['d'])

    success = mean_field_initial_states.initialize_state(psi,args_single)

    if success == -1:
        print('Something is wrong in the initialization...')
        exit(0)

    np.random.seed(index)

    # exit(0)
    dpsi_matter = np.random.uniform(low=-1,high=1,size=psi.matter_shape)
    dpsi_photon = np.random.uniform(low=-1,high=1,size=psi.photon_shape)
    # print(psi.matter)

    # print('\n')
    # print(index)
    # print(psi.matter)

    if noise:
        psi.matter = psi.matter + 1E-4 * dpsi_matter
        psi.photon = psi.photon + 1E-4 * dpsi_photon
        psi.normalize()
    
    # print(dpsi_matter)

    # print(psi.matter)

    # print('\n')
    # exit(0)

    # parameters 

    hn = parameters.local_field_d_levels( psi.matter_shape , args['L'], args['d'], args['W'])
    dt = 1/(args_single['dt_ratio']*np.abs(args_single['omega']))
    args_single['number_measures'] = min(args_single['number_measures'],int(args_single['T']/dt))

    
    # create folder and csv fiargs_singlele for storing and keepin track data

    if save_data:
        folder , success = data_management.create_directory_and_csv(main_dir,f"mean_field_SU{psi.d}_{args_single['initial_state']}",list(args_single.keys()))
        exist = data_management.check_existence_data(folder,args_single)
        index = data_management.get_index_csv(folder,args_single)

        if exist == 1:
            print('Directory exist')
            exit(0)

        elif exist == -1:
            print('Directory does not exist, but header do not match the key')
            exit(0)

        elif exist == 0:
            print('Directory does not exist and headers match. Start the simulation.')

    # changing args for performing time evolution

    args_csv = args_single.copy()

    args_single['hn'] = hn
    args_single['g']  = g
    args_single['dt'] = dt

    # time evolution
    
    obs , times = perform_evolution_and_measure(psi,beta_functions.beta_function_BiC_SUN_mean_field_RWA,args_single,measure_population=measure_population)

    # save data

    if save_data:
        folder , _ = data_management.save_data(folder,'data','mean_field_SU',index,args_single['d'],args_csv,args_json, obs,times)

    return [psi, obs, times]












# def SUN_gaussian_dynamics(args,index,noise=True,save_data=False,main_dir='',measure_population=False):

#     args_json = args.copy()
#     args_single = args.copy()
    

#     alpha = np.array([float(y) for y in args['alpha']])        
#     for k , _ in enumerate(alpha):
#         args_single[f'alpha{k}'] = alpha[k]
#     args_single.pop('alpha')

#     g  = parameters.photon_matter_couplings(alpha)
#     g *= np.sqrt(args_single['omega'])

#     # initial state

#     psi = state.State_bosonic_gaussian_state(args_single['L'],args_single['d'])

#     success = gaussian_initial_states.initialize_state(psi,args_single)

#     if success == -1:
#         print('Something is wrong in the initialization...')
#         exit(0)

#     if noise:
#         np.random.seed(index)
#         dpsi_matter = np.random.uniform(low=-1,high=1,size=psi.matter_shape)
#         dpsi_photon = np.random.uniform(low=-1,high=1,size=psi.photon_shape)
#         psi.matter = psi.matter + 1E-5 * dpsi_matter
#         psi.photon = psi.photon + 1E-5 * dpsi_photon
#         psi.normalize()
    
    
#     # parameters 

#     hn = parameters.local_field_d_levels( psi.matter_shape , args['L'], args['d'], args['W'])
#     dt = 1/(args['dt_ratio']*args['omega'])
#     args['number_measures'] = min(args['number_measures'],int(args['T']/dt))
#     g_matrix = np.zeros(psi.matter_shape,dtype=np.dtype(np.complex128))   
#     for j in range(args['L']):
#         g_matrix[:,:,j] += np.diag(g,k=1)


#     # create folder and csv file for storing and keepin track data

#     if save_data:
#         folder , success = data_management.create_directory_and_csv(main_dir,f"mean_field_SU{psi.d}_{args_single['initial_state']}",list(args_single.keys()))
#         exist = data_management.check_existence_data(folder,args_single)
#         index = data_management.get_index_csv(folder,args_single)

#         if exist == 1:
#             print('Directory exist')
#             exit(0)

#         elif exist == -1:
#             print('Directory does not exist, but header do not match the key')
#             exit(0)

#         elif exist == 0:
#             print('Directory does not exist and headers match. Start the simulation.')

#     # changing args for performing time evolution

#     args_csv = args_single.copy()

#     args_single['hn'] = hn
#     args_single['g']  = g_matrix
#     args_single['dt'] = dt

#     # time evolution
    
#     obs , times = perform_evolution_and_measure(psi,beta_functions.beta_function_BiC_SUN_gaussian_RWA,args_single,measure_population=measure_population)

#     # save data

#     if save_data:
#         folder , _ = data_management.save_data(folder,'data','gaussian_SU',index,args_single['d'],args_csv,args_json, obs,times)

#     return [psi, obs, times]





def SUN_fully_connected_mean_field_dynamics(args,index,noise=True,save_data=False,main_dir='',measure_population=False):

    args_json = args.copy()
    args_single = args.copy()
    

    alpha = np.array([float(y) for y in args['alpha']])        
    for k , _ in enumerate(alpha):
        args_single[f'alpha{k}'] = alpha[k]
    args_single.pop('alpha')

    g  = parameters.photon_matter_couplings(alpha)
    g *= np.sqrt(args_single['omega'])

    g_matrix  = np.zeros((args['d'],args['d']),dtype=np.dtype(np.complex128))   
    g_matrix += np.diag(g,k=1)

    for i in range(2,args['d']):
        g_matrix += np.diag(np.full(shape=args['d']-i,fill_value=args['delta_g']), k=i)


    # initial state

    psi = state.State_bosonic_coherent_state(args_single['L'],args_single['d'])

    success = mean_field_initial_states.initialize_state(psi,args_single)

    if success == -1:
        print('Something is wrong in the initialization...')
        exit(0)

    np.random.seed(index)

    dpsi_matter = np.random.uniform(low=-1,high=1,size=psi.matter_shape)
    dpsi_photon = np.random.uniform(low=-1,high=1,size=psi.photon_shape)

    if noise:
        psi.matter = psi.matter + 1E-4 * dpsi_matter
        psi.photon = psi.photon + 1E-4 * dpsi_photon
        psi.normalize()
    
    # parameters 

    hn = parameters.local_field_d_levels( psi.matter_shape , args['L'], args['d'], args['W'])
    dt = 1/(args_single['dt_ratio']*np.abs(args_single['omega']))
    args_single['number_measures'] = min(args_single['number_measures'],int(args_single['T']/dt))

    
    # create folder and csv fiargs_singlele for storing and keepin track data

    if save_data:
        folder , success = data_management.create_directory_and_csv(main_dir,f"mean_field_SU{psi.d}_{args_single['initial_state']}",list(args_single.keys()))
        exist = data_management.check_existence_data(folder,args_single)
        index = data_management.get_index_csv(folder,args_single)

        if exist == 1:
            print('Directory exist')
            exit(0)

        elif exist == -1:
            print('Directory does not exist, but header do not match the key')
            exit(0)

        elif exist == 0:
            print('Directory does not exist and headers match. Start the simulation.')

    # changing args for performing time evolution

    args_csv = args_single.copy()

    args_single['hn'] = hn
    args_single['g']  = g_matrix
    args_single['dt'] = dt

    # time evolution
    
    obs , times = perform_evolution_and_measure(psi,beta_functions.beta_function_BiC_SUN_mean_field_fully_connected_RWA,args_single,measure_population=measure_population)

    # save data

    if save_data:
        folder , _ = data_management.save_data(folder,'data','fully_connected_mean_field_SU',index,args_single['d'],args_csv,args_json, obs,times)

    return [psi, obs, times]







def SUN_gaussian_dynamics(args,index,noise=True,save_data=False,main_dir='',measure_population=False,connected=0,measure_connected=0):

    args_json = args.copy()
    args_single = args.copy()
    

    alpha = np.array([float(y) for y in args['alpha']])        
    for k , _ in enumerate(alpha):
        args_single[f'alpha{k}'] = alpha[k]
    args_single.pop('alpha')

    g  = parameters.photon_matter_couplings(alpha)
    if args['d'] == 2:
        g = np.array([g[0]])
    g *= np.sqrt(args_single['omega'])

    # initial state

    if connected == 0:
        psi = state.State_bosonic_gaussian_state(args_single['L'],args_single['d'])
        beta_function = beta_functions.beta_function_BiC_SUN_gaussian_RWA
        hn = parameters.local_field_d_levels( psi.matter_shape , args['L'], args['d'], args['W'])
        g_matrix = np.zeros(psi.matter_shape,dtype=np.dtype(np.complex128))   
        

    elif connected == 1:
        psi = state.State_bosonic_gaussian_state_with_amplitudes(args_single['L'],args_single['d'])
        beta_function = beta_functions.beta_function_BiC_SUN_gaussian_with_ampltudes_RWA
        hn = parameters.local_field_d_levels( psi.matter_correlation_shape , args['L'], args['d'], args['W'])
        g_matrix = np.zeros(psi.matter_correlation_shape,dtype=np.dtype(np.complex128))   
        
    # g_matrix = np.zeros((args['d'],args['d']),dtype=np.dtype(np.complex128)) 
    # g_matrix += np.diag(g,k=1)
    # print(g)
    for j in range(args['L']):
        g_matrix[:,:,j] += np.diag(g,k=1)
    

    success = gaussian_initial_states.initialize_state(psi,args_single,connected=connected)

    if success == -1:
        print('Something is wrong in the initialization...')
        exit(0)

    if noise:
        np.random.seed(index)

        if connected == 0:
            dpsi_matter = np.random.uniform(low=-1,high=1,size=psi.matter_shape)
            dpsi_photon = np.random.uniform(low=-1,high=1,size=psi.photon_shape)
            # psi.matter = psi.matter + 1E-8 * dpsi_matter
            psi.photon = psi.photon + 1E-8 * dpsi_photon
            for j in range(args['L']):
                psi.matter[:,:,j] += 1E-8 * (dpsi_matter[:,:,0]/np.abs(np.sum(dpsi_matter[:,:,0])))
            
            psi.normalize()


        elif connected == 1:
            dpsi_ampl_matter = np.random.uniform(low=-1,high=1,size=psi.matter_amplitude_shape)
            dpsi_corr_matter = np.random.uniform(low=-1,high=1,size=psi.matter_correlation_shape)
            dpsi_photon = np.random.uniform(low=-1,high=1,size=psi.photon_shape)
            psi.matter_amplitude = psi.matter_amplitude + 1E-5 * dpsi_ampl_matter
            psi.matter_correlation = psi.matter_correlation + 1E-5 * dpsi_corr_matter

            psi.normalize()

    # parameters 

    dt = 1/(args['dt_ratio']*np.abs(args['omega']))
    args['number_measures'] = min(args['number_measures'],int(args['T']/dt))
    
    # print(dt)

    # exit(0)
    # create folder and csv file for storing and keepin track data

    if save_data:
        folder , success = data_management.create_directory_and_csv(main_dir,f"Gaussian_SU{psi.d}_{args_single['initial_state']}",list(args_single.keys()))
        exist = data_management.check_existence_data(folder,args_single)
        index = data_management.get_index_csv(folder,args_single)

        if exist == 1:
            print('Directory exist')
            exit(0)

        elif exist == -1:
            print('Directory does not exist, but header do not match the key')
            exit(0)

        elif exist == 0:
            print('Directory does not exist and headers match. Start the simulation.')

    # changing args for performing time evolution

    args_csv = args_single.copy()

    args_single['hn'] = hn
    args_single['g']  = g_matrix
    args_single['dt'] = dt

    # time evolution
    
    print('Entering perform_evolution_and_measure')

    obs , times = perform_evolution_and_measure(psi,beta_function,args_single,measure_population=measure_population,measure_connected=measure_connected)

    print('Out of perform_evolution_and_measure')
    # save data

    if save_data:
        folder , _ = data_management.save_data(folder,'data','gaussian_SU',index,args_single['d'],args_csv,args_json, obs,times)

    return [psi, obs, times]


