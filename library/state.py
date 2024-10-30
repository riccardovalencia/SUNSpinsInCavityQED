import numpy as np
import math

# L : number of sites
# d : number of levels on each site (number of atomic modes)
# n_photon : number of photon modes

class State_bosonic_coherent_state:

    def __init__(self,L,d,n_photon = 1):

        self.L  = L
        self.d  = d
        self.n_photon = n_photon
        self.matter_shape  = (self.d, self.L)
        self.photon_shape  = (self.n_photon)

        self.matter  = np.zeros(self.matter_shape , dtype=np.complex128)
        self.photon  = np.zeros(self.photon_shape , dtype=np.complex128)


    def update(self,array):


        if len(array) == math.prod(self.matter_shape) + self.n_photon:
            array_matter = np.reshape(array[:self.L*self.d],self.matter_shape)
            array_photon = np.reshape(array[self.L*self.d:],self.photon_shape)
        
            self.matter = array_matter
            self.photon = array_photon
                       
        else:
            print('Error : input array does not exist or wrong length')
    

    def update_single_site(self,array,j):
        if len(array) == self.d and j >=0 and j < self.L:
            self.matter[:,j] = np.reshape(array,(self.d,))
            
        else:
            print(f'Error : wrong array length ({len(array)}) or site index ({j})')
            print('State is not modified')


    def normalize(self):

        # matter_transpose = np.transpose(self.matter)
        # normalize = np.sum(np.square(matter_transpose),axis=1)
        # matter_transpose = matter_transpose / np.sqrt(normalize)
        # self.matter = np.transpose(matter_transpose)
        
        for j in range(self.L):
            norm = np.sum(np.square(self.matter[:,j]))
            self.matter[:,j] /= np.sqrt(norm)
        

    def copy(self):
        return self
        

    def flatten(self):
        return np.append(np.ndarray.flatten(self.matter), np.ndarray.flatten(self.photon))


    def print(self):
        print(self.matter)
        print(self.photon)


    def save(self, folder = ''):
        np.save(f'{folder}mean_field_state_L{self.L}_d{self.d}_nphoton{self.n_photon}.npy',self.flatten())
        
    def measure_coherence_av(self,j,k):
        if j <= self.d and j >= 0 and k <= self.d and k >= 0:
            return np.average(np.conj(self.matter[j])*self.matter[k])
        else:
            print('Error : out of range levels')
            return float('nan')

    def measure_photon_amplitude(self):
        return self.photon[0]


    def measure_matter_amplitude(self,site,m):
        if site >= 0 and site < self.L and m >= 0 and m < self.d:
            return self.matter[m,site]
        else:
            print('Error : index out of range')
            return float('nan')

    def measure_purity(self):
        # rho_one_body = np.average(np.outer(np.conj(self.matter),np.conj(self.matter)),axis=1)
        # rho2 = rho_one_body @ rho_one_body
        # return np.trace(rho2)
        return 1


class State_bosonic_coherent_state_real_imag(State_bosonic_coherent_state):

    def __init__(self, L,d,n_photon = 1):
        State_bosonic_coherent_state.__init__(self,L,d,n_photon )



    def flatten_real_imag(self):
        return np.concatenate((np.ndarray.flatten(np.real(self.matter)),np.ndarray.flatten(np.imag(self.matter)), np.ndarray.flatten(self.photon)))


    def update_real_imag(self,array):


        if len(array) == 2*math.prod(self.matter_shape) + self.n_photon:
            array_matter_real = np.reshape(array[:math.prod(self.matter_shape)],self.matter_shape)
            array_matter_imag = np.reshape(array[math.prod(self.matter_shape):2*math.prod(self.matter_shape)],self.matter_shape)

            array_photon = np.reshape(array[2*math.prod(self.matter_shape):],self.photon_shape)
    
            self.matter = array_matter_real + 1j * array_matter_imag
            self.photon = array_photon
                       
        else:
            print('Error : input array does not exist or wrong length')
    


class State_bosonic_gaussian_state:

    def __init__(self,L,d,n_photon = 1):

        self.L  = L
        self.d  = d
        self.n_photon = n_photon
        self.matter_shape  = (self.d, self.d, self.L)
        self.photon_shape  = (self.n_photon)

        self.matter  = np.zeros(self.matter_shape , dtype=np.complex128)
        self.photon  = np.zeros(self.photon_shape , dtype=np.complex128)


    def update(self,array):


        if len(array) == math.prod(self.matter_shape) + self.n_photon:
            array_matter = np.reshape(array[:math.prod(self.matter_shape)],self.matter_shape)
            array_photon = np.reshape(array[math.prod(self.matter_shape):],self.photon_shape)
        
            self.matter = array_matter
            self.photon = array_photon
                       
        else:
            print('Error : input array does not exist or wrong length')
    

    def update_single_site(self,array,j):
        # print(self.matter[:,:,j])
        # print(np.reshape(array,(self.d,self.d,)))
        # exit(0)
        if array.shape == (self.d,self.d) and j >=0 and j < self.L:
            self.matter[:,:,j] = array
        # if len(array) == self.d * self.d and j >=0 and j < self.L:
            self.matter[:,:,j] = np.reshape(array,(self.d,self.d,))
            
            
        else:
            print(f'Error : wrong array length ({len(array)}) or site index ({j})')
            print('State is not modified')


    def normalize(self):

        for j in range(self.L):
            norm = np.sum(np.diag(self.matter[:,:,j]))
            self.matter[:,:,j] /= norm



    def copy(self):
        return self
        

    def flatten(self):
        return np.append(np.ndarray.flatten(self.matter), np.ndarray.flatten(self.photon))


    def print(self):
        print(self.matter)
        print(self.photon)


    def save(self, folder = ''):
        np.save(f'{folder}gaussian_state_L{self.L}_d{self.d}_nphoton{self.n_photon}.npy',self.flatten())
        
    def measure_coherence_av(self,j,k):
        if j <= self.d and j >= 0 and k <= self.d and k >= 0:
            C_av = np.average(self.matter,axis=2)
            return C_av[j,k]

        else:
            print('Error : out of range levels')
            return float('nan')

    def measure_photon_amplitude(self):
        return self.photon[0]

    def measure_purity(self):
        # rho_one_body = np.average(self.matter,axis=2)
        # rho2 = rho_one_body @ rho_one_body
        rho2 = np.einsum('ijt,jkt->ikt',self.matter,self.matter)
        rho2 = np.einsum('iit,iit->t',rho2,rho2)
        return np.average(rho2)




class State_bosonic_gaussian_state_real_imag(State_bosonic_gaussian_state):

    def __init__(self, L,d,n_photon = 1):
        State_bosonic_gaussian_state.__init__(self,L,d,n_photon )


    def flatten_real_imag(self):
        return np.concatenate((np.ndarray.flatten(np.real(self.matter)),np.ndarray.flatten(np.imag(self.matter)), np.ndarray.flatten(self.photon)))


    def update_real_imag(self,array):


        if len(array) == 2*math.prod(self.matter_shape) + self.n_photon:
            array_matter_real = np.reshape(array[:math.prod(self.matter_shape)],self.matter_shape)
            array_matter_imag = np.reshape(array[math.prod(self.matter_shape):2*math.prod(self.matter_shape)],self.matter_shape)
            array_photon = np.reshape(array[2*math.prod(self.matter_shape):],self.photon_shape)
    
            self.matter = array_matter_real + 1j * array_matter_imag
            self.photon = array_photon
                       
        else:
            print('Error : input array does not exist or wrong length')



    
class State_bosonic_gaussian_state_with_amplitudes:

    def __init__(self,L,d,n_photon = 1):

        self.L  = L
        self.d  = d
        self.n_photon = n_photon
        self.matter_amplitude_shape = (self.d, self.L)
        self.matter_correlation_shape  = (self.d, self.d, self.L)
        self.matter_shape = (self.matter_amplitude_shape , self.matter_correlation_shape )

        self.photon_shape  = (self.n_photon)

        self.matter_amplitude  = np.zeros(self.matter_amplitude_shape , dtype=np.complex128)
        self.matter_correlation = np.zeros(self.matter_correlation_shape , dtype=np.complex128)
        self.photon  = np.zeros(self.photon_shape , dtype=np.complex128)

        self.ampl_size = math.prod(self.matter_amplitude_shape)
        self.corr_size = math.prod(self.matter_correlation_shape)


    def update(self,array):


        if len(array) == self.ampl_size + self.corr_size + self.n_photon:

            array_matter_amplitude = np.reshape(array[:self.ampl_size],self.matter_amplitude_shape)
            array_matter_correlation = np.reshape(array[self.ampl_size:self.ampl_size+self.corr_size],self.matter_correlation_shape)
            array_photon = np.reshape(array[self.ampl_size+self.corr_size:],self.photon_shape)
        
            self.matter_amplitude   = array_matter_amplitude
            self.matter_correlation = array_matter_correlation
            self.photon = array_photon
                       
        else:
            print('Error : input array does not exist or wrong length')
    
    def update_single_site_correlation(self,array,j):
        
        if array.shape == (self.d,self.d) and j >=0 and j < self.L:
            self.matter_correlation[:,:,j] = array
            self.matter_correlation[:,:,j] = np.reshape(array,(self.d,self.d,))
        else:
            print(f'Error : wrong array length ({len(array)}) or site index ({j})')
            print('State is not modified')



    def update_single_site_amplitude(self,array,j):

        if len(array) == self.d and j >=0 and j < self.L:
            self.matter_amplitude[:,j] = np.reshape(array,(self.d,))
            
        else:
            print(f'Error : wrong array length ({len(array)}) or site index ({j})')
            print('State is not modified')



    def normalize(self):

        for j in range(self.L):
            norm = np.sum(np.diag(self.matter_correlation[:,:,j]))
            self.matter[:,:,j] /= norm



    def copy(self):
        return self
        

    def flatten(self):
        return np.append(np.ndarray.flatten(self.matter_amplitude),np.append( np.ndarray.flatten(self.matter_correlation), np.ndarray.flatten(self.photon)))


    def print(self):
        print(self.matter_amplitude)
        print(self.matter_correlation)
        print(self.photon)


    def save(self, folder = ''):
        np.save(f'{folder}gaussian_state_L{self.L}_d{self.d}_nphoton{self.n_photon}.npy',self.flatten())
        
    def measure_coherence_av(self,j,k):
        if j <= self.d and j >= 0 and k <= self.d and k >= 0:
            C_av = np.average(self.matter_correlation,axis=2)
            return C_av[j,k]

        else:
            print('Error : out of range levels')
            return float('nan')


    def measure_connected_coherence_av(self,j,k):
        if j <= self.d and j >= 0 and k <= self.d and k >= 0:
            C_connected = 0
            for l in range(self.L):
                C_connected += self.matter_correlation[j,k,l] - np.conj(self.matter_amplitude[j,l])*self.matter_amplitude[k,l]            
            # TEST - WE COMPUTE THE DISTANCE BETWEEN MODULUS. IT IS NOT A GOOD MEASURE OF CONNECTED PART
                # C_connected += np.abs(self.matter_correlation[j,k,l]) - np.abs(np.conj(self.matter_amplitude[j,l])*self.matter_amplitude[k,l])            


            return C_connected / self.L

        else:
            print('Error :  out of range levels')
            return float('nan')
            
    def measure_photon_amplitude(self):
        return self.photon[0]


    def measure_matter_amplitude(self,site,m):
        if site >= 0 and site < self.L and m >= 0 and m < self.d:
            return self.matter[m,site]
        else:
            print('Error : index out of range')
            return float('nan')

    


    
