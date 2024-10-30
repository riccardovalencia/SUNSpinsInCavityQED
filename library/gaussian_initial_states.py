from re import L
import numpy as np
import math
import state




# gamma_1 -> array contatining the bosonic amplitudes of the first coherent state
# gamma_2 -> array containing the bosonic amplitudes of the second coherent state
# state is |\psi> = 1/A (|gamma_1> + |gamma_2) with A proper normalization constant

def macroscopic_superposition(N, gamma_1, gamma_2,theta, N_infty = 1,connected=1):
    gamma_1 = gamma_1 * np.exp(1j*theta)

    if N_infty == 0:
        gamma_1 = gamma_1 * np.sqrt(N)
        gamma_2 = gamma_2 * np.sqrt(N)
        overlap = np.exp(1j * theta)
        for g_1 , g_2 in zip(gamma_1,gamma_2):
            overlap *= np.exp(-0.5 * ( np.abs(g_1)**2 + np.abs(g_2)**2 - 2 * np.real(np.conj(g_1)*g_2)  ) )
    else:
        overlap = 0

    A = 2 * (1 + np.real(overlap))

    C = np.outer(np.conj(gamma_1),gamma_1) +  np.outer(np.conj(gamma_2),gamma_2) + np.outer(np.conj(gamma_1),gamma_2) * overlap + np.outer(np.conj(gamma_2),gamma_1) * np.conj(overlap)
    
    C /= A

    # if connected == 1:
    #     C = C - (np.outer(np.conj(gamma_1),gamma_1) +  np.outer(np.conj(gamma_2),gamma_2) + np.outer(np.conj(gamma_1),gamma_2) + np.outer(np.conj(gamma_2),gamma_1)) /(A**2)


    if N_infty == 0:
        C /= N

    
    return C + 0j




# def macroscopic_superposition(N, gamma_1, gamma_2,theta, N_infty = 1, connected = 0):

    


#     # for testing
#     # p = 0.1
#     # dp = 0
#     # b1 = np.sqrt(1/3-p+dp)
#     # b0 = np.sqrt(1/3)
#     # bm1 = np.sqrt(1/3+p-dp)

#     # norm = np.sqrt(np.abs(bm1)**2 + np.abs(b0)**2 + np.abs(b1)**2)

#     # b1  /= norm
#     # b0  /= norm
#     # bm1 /= norm

#     # gamma_1 = np.array([bm1,b0,b1])
#     # gamma_2 = np.array([b0,b1,bm1])

    

#     if N_infty == 0:
#         gamma_1 = gamma_1 * np.sqrt(N)
#         gamma_2 = gamma_2 * np.sqrt(N)
#         overlap = np.exp(1j * theta)
#         for g_1 , g_2 in zip(gamma_1,gamma_2):
#             overlap *= np.exp(-0.5 * ( np.abs(g_1)**2 + np.abs(g_2)**2 - 2 * np.conj(g_1)*g_2  ) )
#     else:
#         overlap = 0
    
#     A = 2 * (1 + np.real(overlap))

#     C = np.outer(np.conj(gamma_1),gamma_1) +  np.outer(np.conj(gamma_2),gamma_2) + np.outer(np.conj(gamma_1),gamma_2) * overlap + np.outer(np.conj(gamma_2),gamma_1) * overlap
    
#     C /= A

#     if connected == 1:
#         C = C - (np.outer(np.conj(gamma_1),gamma_1) +  np.outer(np.conj(gamma_2),gamma_2) + np.outer(np.conj(gamma_1),gamma_2) + np.outer(np.conj(gamma_2),gamma_1)) /(A**2)


    
#     if N_infty == 0:
#         C /= N

    


#     return C + 0j





def initialize_state(psi,args,connected=0):

    if args['initial_state'] == 'macroscopic_superposition':
        print(f"Initializing : {args['initial_state']}")
        gamma_1 = np.array([float(y) for y in args['gamma_1']])
        gamma_2 = np.array([float(y) for y in args['gamma_2']])
        
        for k in range(args['d']):
            args[f'gamma_1_{k}'] = gamma_1[k]
        for k in range(args['d']):
            args[f'gamma_2_{k}'] = gamma_2[k]
        args.pop('gamma_1')
        args.pop('gamma_2')
        args.pop('fraction_quantum_states')
        # gamma_1  /= np.sqrt(np.sum(gamma_1**2))
        # gamma_2  /= np.sqrt(np.sum(gamma_2**2))


        # print(gamma_1)
        # print(gamma_2)
        # exit(0)
        # Definition to obtain agreement with mean field at the level of bosonic coherent states. 
        # In particular, to obtain the same results regarding classical chaos induced via 
        # quantum/classical fluctuactions. 
        # I took inspiration from the mean_field_initial_state
        # -> n_av /= np.sum(n_av)
        # -> beta  = np.sqrt(n_av)

        gamma_1 = np.sqrt(gamma_1/np.sum(gamma_1))
        gamma_2 = np.sqrt(gamma_2/np.sum(gamma_2))

        # print(gamma_1)
        # print(gamma_2)
        # exit(0)
        
        psi_j = macroscopic_superposition(args['N'], gamma_1, gamma_2, args['theta'], args['N_infty'],connected=0)

        if connected == 0:
            for j in range(args['L']):
                psi.update_single_site(psi_j,j)

        elif connected ==1:
            psi_amplitude_j = (gamma_1 + gamma_2)/2
            for j in range(args['L']):
                psi.update_single_site_amplitude(psi_amplitude_j,j)
                psi.update_single_site_correlation(psi_j,j)

        # print(psi.flatten())
        # exit(0)

    elif args['initial_state'] == 'quantum_bubble':
        print(f"Initializing : {args['initial_state']}")
        gamma_1 = np.array([float(y) for y in args['gamma_1']])
        gamma_2 = np.array([float(y) for y in args['gamma_2']])
        
        for k in range(args['d']):
            args[f'gamma_1_{k}'] = gamma_1[k]
        for k in range(args['d']):
            args[f'gamma_2_{k}'] = gamma_2[k]
        args.pop('gamma_1')
        args.pop('gamma_2')

        cat_sites = int(args['L']*args['fraction_quantum_states'])

        
        gamma_1 = np.sqrt(gamma_1/np.sum(gamma_1))
        gamma_2 = np.sqrt(gamma_2/np.sum(gamma_2))

        psi_j_cat = macroscopic_superposition(args['N'], gamma_1, gamma_2, args['theta'], args['N_infty'],connected=0)
        psi_j_coherent = macroscopic_superposition(args['N'], gamma_1, gamma_1, 0, 1,connected=0)

        sites = [j for j in range(args['L'])]

        for _ in range(cat_sites):
            index = np.random.randint(low=0,high=len(sites))
            psi.update_single_site(psi_j_cat,sites[index])
            print(f'cat in {sites[index]}')

            sites.pop(index)
        for j in sites:
            print(f'SUN in {j}')
            psi.update_single_site(psi_j_coherent,j)

    else:
        print(f"Initial state {args['initial_state']} not implemented...")
        exit(0)
    return 1 








# def initialize_state(psi,args,noise,index):

#     if args['initial_state'] == 'macroscopic_superposition':
#         print(f"Initializing : {args['initial_state']}")
#         gamma_1 = np.array([float(y) for y in args['gamma_1']])
#         gamma_2 = np.array([float(y) for y in args['gamma_2']])
        
#         for k in range(args['d']):
#             args[f'gamma_1_{k}'] = gamma_1[k]
#         for k in range(args['d']):
#             args[f'gamma_2_{k}'] = gamma_2[k]
#         args.pop('gamma_1')
#         args.pop('gamma_2')

#         # gamma_1  /= np.sqrt(np.sum(gamma_1**2))
#         # gamma_2  /= np.sqrt(np.sum(gamma_2**2))

#         # print(gamma_1)
#         # print(gamma_2)
#         # exit(0)
#         # Definition to obtain agreement with mean field at the level of bosonic coherent states. 
#         # In particular, to obtain the same results regarding classical chaos induced via 
#         # quantum/classical fluctuactions. 
#         # I took inspiration from the mean_field_initial_state
#         # -> n_av /= np.sum(n_av)
#         # -> beta  = np.sqrt(n_av)

#         if noise:
#             np.random.seed(index)
#             dgamma_1 = np.random.uniform(low=-1,high=1,size=len(gamma_1))
#             dgamma_2 = np.random.uniform(low=-1,high=1,size=len(gamma_1))
#             gamma_1 += 1E-5 * dgamma_1
#             gamma_2 += 1E-5 * dgamma_2

                         
#         gamma_1  /= np.sqrt(np.sum(gamma_1**2))
#         gamma_2  /= np.sqrt(np.sum(gamma_2**2))

#         # print(gamma_1)
#         # print(gamma_2)
#         # exit(0)
        
#         psi_j = macroscopic_superposition(args['N'], gamma_1, gamma_2, args['theta'], args['N_infty'])
#         # print(psi_j)
#         for j in range(args['L']):
#             psi.update_single_site(psi_j,j)
#         # print(psi.matter)
    
#     else:
#         print(f"Initial state {args['initial_state']} not implemented...")
#         exit(0)
#     return 1 