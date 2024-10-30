import numpy as np
import math
import opt_einsum

# A beta function beta is defined as being the 'velocity' of the variables y.
# Specifically \dot{y} = beta(y) (not necessarilty linear in y).

def beta_function_BiC_SUN_mean_field_RWA(t, state, args):

    args_param = args[0]
    matter_shape , photon_shape = args[1]

    omega = args_param['omega']
    k     = args_param['k']
    g     = args_param['g']
    hn    = args_param['hn']

    beta_matter = np.zeros(matter_shape,dtype=np.dtype(np.complex128))    
    beta_photon = np.zeros(photon_shape,dtype=np.dtype(np.complex128))    

    matter = np.reshape(state[:matter_shape[0]*matter_shape[1]], matter_shape)
    photon = np.reshape(state[ matter_shape[0]*matter_shape[1]:], photon_shape)

    delta = 0 
    for n in range(matter_shape[0]-1):
        delta += g[n] * np.average(np.conj(matter[n]) *matter[n+1] )

    if args_param['adiabatic_elimination'] == 1:
        photon = -delta/omega

    else:
        beta_photon[0] = - (1j * omega + k/2) * photon -1j * delta

    beta_matter[0] = -1j * hn[0] * matter[0] - 1j * g[0] * matter[1] * np.conj(photon)
    for j in range(1,matter_shape[0]-1):
        beta_matter[j] = - 1j * hn[j]  * matter[j]  - 1j * g[j]  * matter[j+1] * np.conj(photon) - 1j * g[j-1] * matter[j-1] * photon
    beta_matter[-1]    = - 1j * hn[-1] * matter[-1] - 1j * g[-1] * matter[-2]  * photon 




    return np.append( np.ndarray.flatten(beta_matter) , beta_photon )



def beta_function_BiC_SUN_mean_field_RWA_real_imag(t, state, args):

    args_param = args[0]
    matter_shape , photon_shape = args[1]

    omega = args_param['omega']
    k     = args_param['k']
    g     = args_param['g']
    hn    = args_param['hn']

    beta_matter = np.zeros(matter_shape,dtype=np.dtype(np.complex128)) 
    beta_photon = np.zeros(photon_shape)

    matter_real = np.reshape(state[:math.prod(matter_shape)], matter_shape) 
    matter_imag = np.reshape(state[math.prod(matter_shape):2*math.prod(matter_shape)], matter_shape) 
    photon = np.reshape(state[ 2*math.prod(matter_shape):], photon_shape)

    matter = matter_real + 1j * matter_imag

    delta = 0 
    for n in range(matter_shape[0]-1):
        delta += g[n] * np.average(np.conj(matter[n]) *matter[n+1] )

    if args_param['adiabatic_elimination'] == 1:
        photon = -delta/omega

    else:
        beta_photon[0] = - (1j * omega + k/2) * photon -1j * delta

    beta_matter[0] = -1j * hn[0] * matter[0] - 1j * g[0] * matter[1] * np.conj(photon)
    for j in range(1,matter_shape[0]-1):
        beta_matter[j] = - 1j * hn[j]  * matter[j]  - 1j * g[j]  * matter[j+1] * np.conj(photon) - 1j * g[j-1] * matter[j-1] * photon
    beta_matter[-1]    = - 1j * hn[-1] * matter[-1] - 1j * g[-1] * matter[-2]  * photon 


    beta_matter_real = np.real(beta_matter)
    beta_matter_imag = np.imag(beta_matter)
    
    return np.concatenate( (np.ndarray.flatten(beta_matter_real),np.ndarray.flatten(beta_matter_imag) , beta_photon ))


def beta_function_BiC_SUN_Gaussian_RWA_real_imag(t, state, args):

    args_param = args[0]
    matter_shape , photon_shape = args[1]

    L = args_param['L']
    omega = args_param['omega']
    k     = args_param['k']
    g     = args_param['g']
    hn    = args_param['hn']


    beta_matter = np.zeros(matter_shape,dtype=np.dtype(np.complex128))    
    beta_photon = np.zeros(photon_shape) 


    matter_real = np.reshape(state[:math.prod(matter_shape)], matter_shape) 
    matter_imag = np.reshape(state[math.prod(matter_shape):2*math.prod(matter_shape)], matter_shape) 
    photon = np.reshape(state[ 2*math.prod(matter_shape):], photon_shape)

    matter = matter_real + 1j * matter_imag

    C_av = np.average(matter,axis=2)

    delta = np.sum(np.diag(g[:,:,0],k=1)*np.diag(C_av,k=-1))
            
    if args_param['adiabatic_elimination'] == 1:
        photon = -delta/omega

    else:
        beta_photon[0] = - (1j * omega + k/2) * photon -1j * delta

    H = hn + g * np.conj(photon) + np.transpose(g,axes=(1,0,2)) * photon

    beta_matter = 1j * ( opt_einsum.contract('ijt,jkt->ikt',H,matter) -  opt_einsum.contract('ijt,jkt->ikt',matter,H)  )

    beta_matter_real = np.real(beta_matter)
    beta_matter_imag = np.imag(beta_matter)
    
    
    return np.concatenate( (np.ndarray.flatten(beta_matter_real),np.ndarray.flatten(beta_matter_imag) , beta_photon ))




# the Hamiltonian is of the form H = \sum_{i,j} h_{i,j} b_i^\dagger b_j, where
# h_{i,i} = h_i
# h_{i+1,i} = g_i a
# where the photon a can be enslaved or not
# We have 
# d\Sigma/dt = i[h,\Sigma]
# where \Sigma_{ij} = b_i^\dagger b_j

# deprecated in favour of a more optimized contraction based on opt_einsum (see below)

# def beta_function_BiC_SUN_gaussian_with_ampltudes_RWA(t, state, args):

    
#     args_param = args[0]
#     matter_shape , photon_shape = args[1]

#     amplitude_shape   = matter_shape[0]
#     correlation_shape = matter_shape[1]

#     L = args_param['L']
#     omega = args_param['omega']
#     k     = args_param['k']
#     g     = args_param['g']
#     hn    = args_param['hn']


#     beta_matter_amplitude   = np.zeros(amplitude_shape,dtype=np.dtype(np.complex128))   
#     beta_matter_correlation = np.zeros(correlation_shape,dtype=np.dtype(np.complex128))   
#     beta_photon = np.zeros(photon_shape,dtype=np.dtype(np.complex128))    

#     ampl_size = math.prod(amplitude_shape)
#     corr_size = math.prod(correlation_shape)

#     matter_amplitude = np.reshape(state[:ampl_size], amplitude_shape)
#     matter_correlation = np.reshape(state[ampl_size:ampl_size+corr_size], correlation_shape)
#     photon = np.reshape(state[ampl_size+corr_size:], photon_shape)

#     # C_av = np.average(matter_correlation,axis=2)
#     # delta = np.sum(np.diag(g[:,:,0],k=1)*np.diag(C_av,k=-1))
        
#     # if args_param['adiabatic_elimination'] == 1:
#     #     photon = -delta/omega

#     # else:
#     #     beta_photon[0] = - (1j * omega + k/2) * photon -1j * delta



#     # g_amp = np.diag(g[:,:,0],k=1)
#     g_amp = np.diag(g,k=1)
#     h_amp = np.diag(hn[:,:,0],k=0)

#     delta = 0 
#     for n in range(amplitude_shape[0]-1):
#         delta += g_amp[n] * np.average(np.conj(matter_amplitude[n]) *matter_amplitude[n+1] )

#     if args_param['adiabatic_elimination'] == 1:
#         photon = -delta/omega

#     else:
#         beta_photon[0] = - (1j * omega + k/2) * photon -1j * delta


#     beta_matter_amplitude[0] = -1j * h_amp[0] * matter_amplitude[0] - 1j * g_amp[0] * matter_amplitude[1] * np.conj(photon)
#     for j in range(1,amplitude_shape[0]-1):
#         beta_matter_amplitude[j] = - 1j * h_amp[j]  * matter_amplitude[j]  - 1j * g_amp[j]  * matter_amplitude[j+1] * np.conj(photon) - 1j * g_amp[j-1] * matter_amplitude[j-1] * photon
#     beta_matter_amplitude[-1]    = - 1j * h_amp[-1] * matter_amplitude[-1] - 1j * g_amp[-1] * matter_amplitude[-2]  * photon 

#     C_av = np.average(matter_correlation,axis=2)
#     delta = np.sum(np.diag(g[:,:,0],k=1)*np.diag(C_av,k=-1))
        
#     if args_param['adiabatic_elimination'] == 1:
#         photon = -delta/omega

#     else:
#         beta_photon[0] = - (1j * omega + k/2) * photon -1j * delta

#     for j in range(L):
#         Hj = hn[:,:,j] + g * np.conj(photon) + np.transpose(g) * photon
#         beta_matter_correlation[:,:,j] = 1j * (Hj @ matter_correlation[:,:,j] - matter_correlation[:,:,j] @ Hj)

   

#     return np.append( np.ndarray.flatten(beta_matter_amplitude),np.append( np.ndarray.flatten(beta_matter_correlation) , beta_photon ))





def beta_function_BiC_SUN_gaussian_RWA_triangular(t, state, args):

    
    args_param = args[0]
    matter_shape , photon_shape = args[1]

    L = args_param['L']
    omega = args_param['omega']
    k     = args_param['k']
    g     = args_param['g']
    hn    = args_param['hn']


    beta_matter = np.zeros(matter_shape,dtype=np.dtype(np.complex128))    
    beta_photon = np.zeros(photon_shape,dtype=np.dtype(np.complex128))    

    matter = np.reshape(state[:math.prod(matter_shape)], matter_shape)
    photon = np.reshape(state[math.prod(matter_shape):], photon_shape)

    C_av = np.average(matter,axis=2)
    delta = np.sum(np.diag(g[:,:,0],k=1)*np.diag(C_av,k=-1))
        
    if args_param['adiabatic_elimination'] == 1:
        photon = -delta/omega

    else:
        beta_photon[0] = - (1j * omega + k/2) * photon -1j * delta

    # Hj = hn + 1j * np.zeros(matter_shape,dtype=np.dtype(np.complex128))  
    # Hj = hn + g * photon + np.transpose(g[:,:,j]) * np.conj(photon)
    
    # Hj += g*photon + np.transpose(g)*np.conj(photon)

    for j in range(L):
        Hj = hn[:,:,j] + g[:,:,j] * np.conj(photon) + np.transpose(g[:,:,j]) * photon
        # print(Hj)
        # exit(0)
        # Hj[:,:,j] += g[:,:,j] * photon + np.transpose(g[:,:,j]) * np.conj(photon)
        # beta_matter[:,:,j] = 1j * (Hj[:,:,j] @ matter[:,:,j] - matter[:,:,j] @ Hj[:,:,j])
        beta_matter[:,:,j] = 1j * (Hj @ matter[:,:,j] - matter[:,:,j] @ Hj)

    # beta_matter = np.array([ 1j * ((hn[:,:,j] + g[:,:,j] * np.conj(photon) + np.transpose(g[:,:,j]) * photon) @ matter[:,:,j] 
    #             - matter[:,:,j] @ (hn[:,:,j] + g[:,:,j] * np.conj(photon) + np.transpose(g[:,:,j]) * photon)) for j in range(L)])

    # beta_matter = np.array(beta_matter)

    return np.append( np.ndarray.flatten(beta_matter) , beta_photon )





def beta_function_BiC_SUN_mean_field_fully_connected_RWA(t, state, args):

    args_param = args[0]
    matter_shape , photon_shape = args[1]

    L = args_param['L']
    d = args_param['d']
    omega = args_param['omega']
    k     = args_param['k']
    g     = args_param['g'].copy()
    hn    = args_param['hn']

    beta_matter = np.zeros(matter_shape,dtype=np.dtype(np.complex128))    
    beta_photon = np.zeros(photon_shape,dtype=np.dtype(np.complex128))    

    matter = np.reshape(state[:matter_shape[0]*matter_shape[1]], matter_shape)
    photon = np.reshape(state[ matter_shape[0]*matter_shape[1]:], photon_shape)

    
    delta = 0  
    for j in range(L):
        delta += np.conj(matter[:,j].T) @ g @ matter[:,j]
    delta /= L


    if args_param['adiabatic_elimination'] == 1:
        photon = -delta/omega

    else:
        beta_photon[0] = - (1j * omega + k/2) * photon -1j * delta

    for n in range(d):
        for k in range(n+1,d):
            g[n,k] *= np.conj(photon)


    for j in range(L):
        Hj = -1j* ( np.diag(hn[:,j]) + g + np.conj(np.transpose(g)))
        beta_matter[:,j] = Hj @ matter[:,j]


    return np.append( np.ndarray.flatten(beta_matter) , beta_photon )




def beta_function_BiC_SUN_mean_field_Z2(t, state, args):

    args_param = args[0]
    matter_shape , photon_shape = args[1]

    omega = args_param['omega']
    k     = args_param['k']
    g     = args_param['g']
    hn    = args_param['hn']

    beta_matter = np.zeros(matter_shape,dtype=np.dtype(np.complex128))    
    beta_photon = np.zeros(photon_shape,dtype=np.dtype(np.complex128))    

    matter = np.reshape(state[:matter_shape[0]*matter_shape[1]], matter_shape)
    photon = np.reshape(state[ matter_shape[0]*matter_shape[1]:], photon_shape)

    delta = 0 
    for n in range(matter_shape[0]-1):
        delta += 2*g[n] * np.average(np.real(np.conj(matter[n]) *matter[n+1]) )

    if args_param['adiabatic_elimination'] == 1:
        photon = -delta/omega

    else:
        beta_photon[0] = - (1j * omega + k/2) * photon -1j * delta

    beta_matter[0] = -1j * hn[0] * matter[0] - 1j * g[0] * matter[1] * 2 * np.real(photon)
    for j in range(1,matter_shape[0]-1):
        beta_matter[j] = - 1j * hn[j]  * matter[j]  - 1j * g[j]  * matter[j+1] * 2 * np.real(photon) - 1j * g[j-1] * matter[j-1] * 2 * np.real(photon)
    beta_matter[-1]    = - 1j * hn[-1] * matter[-1] - 1j * g[-1] * matter[-2]  * 2 * np.real(photon) 




    return np.append( np.ndarray.flatten(beta_matter) , beta_photon )




# optimized via einstein summation np.einsum of 
# beta_function_BiC_SUN_gaussian_RWA


def beta_function_BiC_SUN_gaussian_RWA(t, state, args):

    
    args_param = args[0]
    matter_shape , photon_shape = args[1]

    L = args_param['L']
    omega = args_param['omega']
    k     = args_param['k']
    g     = args_param['g']
    hn    = args_param['hn']


    beta_matter = np.zeros(matter_shape,dtype=np.dtype(np.complex128))    
    beta_photon = np.zeros(photon_shape,dtype=np.dtype(np.complex128))    

    matter = np.reshape(state[:math.prod(matter_shape)], matter_shape)
    photon = np.reshape(state[math.prod(matter_shape):], photon_shape)

    C_av = np.average(matter,axis=2)

    delta = np.sum(np.diag(g[:,:,0],k=1)*np.diag(C_av,k=-1))
    
    # delta = np.sum(np.diag(g,k=1)*np.diag(C_av,k=-1))
        
    if args_param['adiabatic_elimination'] == 1:
        photon = -delta/omega

    else:
        beta_photon[0] = - (1j * omega + k/2) * photon -1j * delta

    H = hn + g * np.conj(photon) + np.transpose(g,axes=(1,0,2)) * photon
    # print(H[:,:,0])
    # exit(0)
    beta_matter = 1j * ( opt_einsum.contract('ijt,jkt->ikt',H,matter) -  opt_einsum.contract('ijt,jkt->ikt',matter,H)  )

    ### benchmark with known exact method. Max element of the distance of the two beta_functions is 0. Thus, they are equivalent.

    # beta_matter_opt = 1j * ( opt_einsum.contract('ijt,jkt->ikt',H,matter) -  opt_einsum.contract('ijt,jkt->ikt',matter,H)  )

    # for j in range(L):
    #     Hj = hn[:,:,j] + g[:,:,j] * np.conj(photon) + np.transpose(g[:,:,j]) * photon
    #     beta_matter[:,:,j] = 1j * (Hj @ matter[:,:,j] - matter[:,:,j] @ Hj)
    # print('distance')
    # print(np.amax(beta_matter-beta_matter_opt))
    # print(beta_matter_opt.shape)
    # print(beta_matter.shape)
    # exit(0)
    # beta_matter = np.array([ 1j * ((hn[:,:,j] + g[:,:,j] * np.conj(photon) + np.transpose(g[:,:,j]) * photon) @ matter[:,:,j] 
    #             - matter[:,:,j] @ (hn[:,:,j] + g[:,:,j] * np.conj(photon) + np.transpose(g[:,:,j]) * photon)) for j in range(L)])

    # beta_matter = np.array(beta_matter)

    return np.append( np.ndarray.flatten(beta_matter) , beta_photon )
