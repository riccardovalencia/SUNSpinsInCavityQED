import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(f'./library')
import state
import parameters


def macroscopic_superposition(N, gamma_1, gamma_2,theta, N_infty = 1,connected=1):

    if N_infty == 0:
        gamma_1 = gamma_1 * np.sqrt(N)
        gamma_2 = gamma_2 * np.sqrt(N)
        overlap = np.exp(1j * theta)
        for g_1 , g_2 in zip(gamma_1,gamma_2):
            overlap *= np.exp(-0.5 * ( np.abs(g_1)**2 + np.abs(g_2)**2 - 2 * np.conj(g_1)*g_2  ) )
    else:
        overlap = 0
    
    A = 2 * (1 + np.real(overlap))

    C = np.outer(np.conj(gamma_1),gamma_1) +  np.outer(np.conj(gamma_2),gamma_2) + np.outer(np.conj(gamma_1),gamma_2) * overlap + np.outer(np.conj(gamma_2),gamma_1) * overlap
    
    C /= A


    # if connected == 1:
    #     C = C - (np.outer(np.conj(gamma_1),gamma_1) +  np.outer(np.conj(gamma_2),gamma_2) + np.outer(np.conj(gamma_1),gamma_2) + np.outer(np.conj(gamma_2),gamma_1)) /(A**2)


    if N_infty == 0:
        C /= N

    return C + 0j



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

        

        gamma_1 = np.sqrt(gamma_1/np.sum(gamma_1))
        gamma_2 = np.sqrt(gamma_2/np.sum(gamma_2))

       
        psi_j = macroscopic_superposition(args['N'], gamma_1, gamma_2, args['theta'], args['N_infty'],connected=0)

        if connected == 0:
            for j in range(args['L']):
                psi.update_single_site(psi_j,j)

        elif connected ==1:
            psi_amplitude_j = (gamma_1 + gamma_2)/2
            for j in range(args['L']):
                psi.update_single_site_amplitude(psi_amplitude_j,j)
                psi.update_single_site_correlation(psi_j,j)

        
        
    else:
        print(f"Initial state {args['initial_state']} not implemented...")
        exit(0)
    return 1 


p_list = [0.29 + 0.0005 * j for j in range(20)]
p_list = np.linspace(0,1/3,num=1000)
p_list = [1/3]
x_list = []
for p in p_list:
    x_list.append([f'{1/3+p:.12f}', f'{1/3:.12f}', f'{1/3 - p:.12f}'])
# for p in p_list:
#     x_list.append([f'{1/2+p:.12f}', f'{1/2 - p:.12f}'])

purity = []
det =  []
args = {}
args['initial_state'] = 'macroscopic_superposition'
args['L'] = 1
args['d'] = 3
args['N'] = 1
args['theta'] = 0
args['N_infty'] = 1
psi = state.State_bosonic_gaussian_state(args['L'],args['d'])
alpha = [0.46]
g = parameters.photon_matter_couplings(alpha)
print(g)
for idx, x in enumerate(x_list):

    args['gamma_1'] = x
    args['gamma_2'] = np.roll(x,-1)

    success = initialize_state(psi,args,connected=0)
    
    rho = psi.matter[:,:,0]
    print(rho)
    rho2  = rho @ rho
    purity.append(np.trace(rho2))
    det.append(np.linalg.det(rho))

fig, ax = plt.subplots(1,1)

# ax.plot(p_list,1-np.abs(purity))
ax.plot(p_list,1-np.abs(det))
ax.set_xscale('log')
ax.set_yscale('log')
print(purity)
plt.show()