import numpy as np
import math
import state

def spin_coherent_state_two_levels(theta,phi):

    hx = np.sin(theta) * np.cos(phi)
    hy = np.sin(theta) * np.sin(phi)
    hz = np.cos(theta)

    H = -1 * np.array([[ hz , hx + 1j * hy ] ,[ hx - 1j * hy , -hz]])

    _ , eigenstates = np.linalg.eigh(H)


    # constant_1 = hx**2 + hy**2 + hz**2
    # constant_2 = constant_1 * (hx**2 + hy**2 + 2 * hz * (hz + np.sqrt(constant_1)))

    # # if g_+ = 0
    # bk = np.sqrt(0.5 - hz/(2.* np.sqrt(hx**2 + hy**2 + hz**2)))
    # b0 = ( bk * (hz + np.sqrt(hx**2 + hy**2 + hz**2))) / (hx + 1j*hy)

    # print(bk)
    # print(b0)

    # print(eigenstates[:,0])

    return eigenstates[:,0]



def spin_coherent_state_three_levels(theta,phi):
    hx = np.sin(theta) * np.cos(phi)
    hy = np.sin(theta) * np.sin(phi)
    hz = np.cos(theta)

    H1 = [-hz , -(hx - 1j * hy)/np.sqrt(2) , 0]
    H2 = [-(hx+1j*hy)/np.sqrt(2) , 0 , -(hx - 1j * hy)/np.sqrt(2)]
    H3 = [0 , -(hx + 1j * hy)/np.sqrt(2) , hz]

    H = np.array([H1,H2,H3])

    _ , eigenstates = np.linalg.eigh(H)

    # check

    # constant_1 = hx**2 + hy**2 + hz**2
    # constant_2 = constant_1 * (hx**2 + hy**2 + 2 * hz * (hz + np.sqrt(constant_1)))
    # bm1 = (hx**2 + hy**2) / (2 * np.sqrt(constant_2))
    # b0  = (hx + 1j * hy) * (hz + np.sqrt(constant_1)) / (np.sqrt(2) * np.sqrt(constant_2))
    # b1  = (hx + 1j * hy) * np.sqrt( 1 + (hz * (hz + 2 * np.sqrt(constant_1))/ constant_1) )/ (2 * (hx - 1j * hy))

    # print(b1)
    # print(b0)
    # print(bm1)


    # print(eigenstates[:,0])

    # return np.array([b1,b0,bm1])

    return eigenstates[:,0]


def spin_coherent_state_three_levels(theta,phi):
    hx = np.sin(theta) * np.cos(phi)
    hy = np.sin(theta) * np.sin(phi)
    hz = np.cos(theta)

    H1 = [-hz , -(hx - 1j * hy)/np.sqrt(2) , 0]
    H2 = [-(hx+1j*hy)/np.sqrt(2) , 0 , -(hx - 1j * hy)/np.sqrt(2)]
    H3 = [0 , -(hx + 1j * hy)/np.sqrt(2) , hz]

    H = np.array([H1,H2,H3])

    _ , eigenstates = np.linalg.eigh(H)

    # check

    # constant_1 = hx**2 + hy**2 + hz**2
    # constant_2 = constant_1 * (hx**2 + hy**2 + 2 * hz * (hz + np.sqrt(constant_1)))
    # bm1 = (hx**2 + hy**2) / (2 * np.sqrt(constant_2))
    # b0  = (hx + 1j * hy) * (hz + np.sqrt(constant_1)) / (np.sqrt(2) * np.sqrt(constant_2))
    # b1  = (hx + 1j * hy) * np.sqrt( 1 + (hz * (hz + 2 * np.sqrt(constant_1))/ constant_1) )/ (2 * (hx - 1j * hy))

    # print(b1)
    # print(b0)
    # print(bm1)


    # print(eigenstates[:,0])

    # return np.array([b1,b0,bm1])

    return eigenstates[:,0]


def sun_coherent_state_four_levels(theta,phi):
    hx = np.sin(theta) * np.cos(phi)
    hy = np.sin(theta) * np.sin(phi)
    hz = np.cos(theta)

    H1 = [0 ,hx,0 ,0]
    H2 = [hx,0 ,hy,0]
    H3 = [0 ,hy,0 ,hz]
    H4 = [0 ,0 ,hz,0]

    H = np.array([H1,H2,H3,H4])

    _ , eigenstates = np.linalg.eigh(H)

    
    return eigenstates[:,0]



def initialize_spin_coherent_state(L,j,theta,phi,kink,spin_d):
        
    if kink == 1:
        if theta < math.pi/2:
            theta_j_pos = theta
            theta_j_neg = theta +  2*(math.pi/2-theta)

        else:
            theta_j_pos = theta - 2*(math.pi/2-theta)
            theta_j_neg = theta 

        if j < L/2:
            theta_j = theta_j_neg
        else:
            theta_j = theta_j_pos

    else:
        theta_j = theta
        

    return spin_d(theta_j,phi)
    



def initialize_state(psi,args):

    if args['initial_state'] == 'spin':
        args.pop('population')
        theta  = args['theta'] * math.pi
        phi    = args['phi']   * math.pi
        if args['d'] == 2:
            spin_d_coherent_state = spin_coherent_state_two_levels

        elif args['d'] == 3:
            spin_d_coherent_state = spin_coherent_state_three_levels

        elif args['d'] == 4:
            spin_d_coherent_state = sun_coherent_state_four_levels

        else:
            print(f"Spin {args['d']} not implemented. Exit.")
            return -1

        for j in range(args['L']):
            psi_j = initialize_spin_coherent_state(args['L'],j,theta,phi,args['kink'],spin_d_coherent_state)
            psi.update_single_site(psi_j,j)
    
    else:
        args.pop('theta')
        args.pop('phi')


    if args['initial_state'] == 'coherent_state':

        if not 'population' in args:
            print("Missing 'population' in args. Exit")
            exit -1

        n_av = np.array([float(y) for y in args['population']])
        for k in range(args['d']):
            args[f'n{k}'] = n_av[k]
        args.pop('population')

    

        n_av /= np.sum(n_av)
        beta  = np.sqrt(n_av)

        for j in range(args['L']):

            if args['kink'] == 0:
                psi_j = beta

            if args['kink'] == 1:
                if j < args['L']/2:
                    psi_j = beta
                else:
                    psi_j = np.flip(beta)

            psi.update_single_site(psi_j,j)

    
    if args['initial_state'] == 'coherent_state_rolled':

        if not 'population' in args:
            print("Missing 'population' in args. Exit")
            exit -1

        n_av = np.array([float(y) for y in args['population']])
        for k in range(args['d']):
            args[f'n{k}'] = n_av[k]
        args.pop('population')

        # n_av /= np.sqrt(np.sum(n_av**2))
        # beta  = n_av

        n_av /= np.sum(n_av)
        beta  = np.sqrt(n_av)

        # beta 

        for j in range(args['L']):

            if args['kink'] == 0:
                psi_j = beta

            if args['kink'] == 1:
                if j < args['L']/2:
                    psi_j = beta
                else:
                    psi_j = np.roll(beta,1)

            psi.update_single_site(psi_j,j)
            


    return 1