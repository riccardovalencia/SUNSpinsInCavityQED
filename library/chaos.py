import numpy as np
from scipy.signal import argrelextrema
import time
import matplotlib.pyplot as plt


def compute_average_lyapunov_exponent(O,t,max_dO=-1,order=10,transient = 1,eps=1E-2, cut_off_distance = 1E-8,min_dO = -8):
    O_keys = list(O.keys())
    N = len(O[O_keys[0]])

    dO = []
    for i in range(N):
        for j in range(i+1,N):

            total_distance = 0
            for name in O_keys:
                total_distance += np.abs(O[name][i][0]-O[name][j][0])

            if total_distance < eps and total_distance > 0:
                
                do_av = 0 
                for name in O_keys:
                    # do_av += np.square(np.abs(O[name][i][transient:]-O[name][j][transient:]))
                    do_av += np.square(np.real(O[name][i][1:]-O[name][j][1:]))
                    do_av += np.square(np.imag(O[name][i][1:]-O[name][j][1:]))
    
                dO.append(np.log(np.sqrt(do_av)/len(O_keys)))

    dO_av = np.average(dO,axis=0)
    dO_st = np.std(dO,axis=0)#/np.sqrt(len(dO))
    t = t[1:]
    # print(dO_av)
    # exit(0)
    index = t > transient
    # index = dO_av < np.log10(max_dO)
    dO_av = dO_av[index]
    dO_st = dO_st[index]
    t = t[index]

    # index =  argrelextrema(dO_av, np.greater,order=order)[0]
    # dO_av = dO_av[index]
    # t = t[index]
    # dO_st = dO_st[index]

    # index = dO_av < max_dO
    index = np.argmax(dO_av > max_dO)
    if index == 0:
        index = len(dO_av)
    dO_av = dO_av[:index]
    dO_st = dO_st[:index]
    t = t[:index]


    index = dO_av > min_dO

    dO_av = dO_av[index]
    dO_st = dO_st[index]
    t = t[index]

    # index = dO_av > -3
    # dO_av = dO_av[index]
    # dO_st = dO_st[index]
    # t = t[index]
    if len(t) > 2:
        [m , _] , C = np.polyfit(t, dO_av, 1, w = 1/dO_st,cov=True)
        [_, _]  , residual_exp , _ , _ , _ = np.polyfit(t, dO_av, 1,w = 1/ dO_st,full=True)
        [_ , _] , residual_pol , _ , _ , _ = np.polyfit(np.log(t), dO_av, 1,w = 1/  dO_st,full=True)
        # print(f'error pol: {resitual_pol}')
        # print(f'error exp: {residual_exp}')

        if residual_exp < residual_pol:
            lyapunov  = m 
            dlyapunov = np.sqrt(C[0,0])
        else:
            lyapunov = 0 
            dlyapunov = 0
    else:

        return [ float('nan'), float('nan')], t , [dO_av, dO_st] ,  float('nan')
    
    return [lyapunov,dlyapunov], t , [dO_av, dO_st] , residual_exp[0]/residual_pol[0]

# def compute_average_lyapunov_exponent(O,t,max_dO=1E-2,transient = 1,eps=1E-2, cut_off_distance = 1E-6):
#     O_keys = list(O.keys())
#     N = len(O[O_keys[0]])

#     dO = {}
    
#     for name in O_keys:
#         dO[name] = []

#     dO_keys = list(dO.keys())
    
#     for i in range(N):
#         for j in range(i+1,N):

#             total_distance = 0
#             for name in O_keys:
#                 total_distance += np.abs(O[name][i][0]-O[name][j][0])

#             if total_distance < eps and total_distance > 0:

#                 for name in dO_keys:
#                     dO[name].append(np.log(np.abs(O[name][i][transient:]-O[name][j][transient:])))
#                     dO[name].append(np.log(np.abs(O[name][i][transient:]-O[name][j][transient:])))
    
#     dO_av = 0
#     dO_st = 0
#     for name in dO_keys:
#         dO[name]  = np.array(dO[name])
#         DeltaO_av = np.average(dO[name],axis=0)
#         dO_av += np.square(DeltaO_av)
#         dO_st += np.square(DeltaO_av) * np.var(dO[name],axis=0) / len(dO[name])

  
#     dO_av = np.sqrt(dO_av)
#     dO_st = np.sqrt(dO_st)
    
#     return [0,0], t , [dO_av, dO_st]
    # index =  argrelextrema(o, np.greater,order=10)[0]

            # o = o[index]
            # t_ = t[index]

            # index = o < max_dO
            # o = o[index]
            # t_ = t_[index]


    
    #     if len(index) > 2 :
    #         # ax[idx].plot(t[index],dO_av[index])
   
    #         # m , q = np.polyfit(t_, dO_av, 1,cov=False)
    #         # m , q = np.polyfit(t[index], o[index], 1,cov=False)
    #         # lyapunov.append(m)

    #         [m , q] , residual_exp , _ , _ , _ = np.polyfit(t_, o, 1, full=True)
    #         # [_ , _] , residual_lin , _ , _ , _ = np.polyfit(t_, np.exp(o), 1, full=True)
    #         # print(residual_lin)
    #         # print(residual_exp)
    #         # time.sleep(1)
    #         # if residual_exp < residual_lin:
    #         lyapunov.append(m)
    #         # else:
    #         #     lyapunov.append(0)


    #     if max_lyapunov < np.average(np.array(lyapunov)):
    #         max_lyapunov = np.average(np.array(lyapunov))

        
    # return max_lyapunov, t, dO


def compute_lyapunov_exponent(O,t,max_dO=1E-2,order=10,transient = 1,eps=1E-2, cut_off_distance = 1E-8):

    O_keys = list(O.keys())
    N = len(O[O_keys[0]])

    dO = {}
    
    for name in O_keys:
        dO[name] = []

    dO_keys = list(dO.keys())
    
    for i in range(N):
        for j in range(i+1,N):

            total_distance = 0
            for name in O_keys:
                total_distance += np.abs(O[name][i][0]-O[name][j][0])

            if total_distance < eps and total_distance > 0:

                for name in dO_keys:
                    dO[name].append(np.log(np.abs(O[name][i][transient:]-O[name][j][transient:])))
    
    
    for name in dO_keys:
        dO[name] = np.array(dO[name])


    lyapunov = []
    dlyapunov = []
    for idx, name in enumerate(dO_keys):
        dO_av = np.average(dO[name],axis=0)
        dO_st = np.std(dO[name],axis=0) / np.sqrt(len(dO[name]))
   
        # index = np.where(dO_av > np.log10(cut_off_distance))[0]

        # index = argrelextrema(dO_av, np.greater,order=int(len(t)/30))[0]

        index =  argrelextrema(dO_av, np.greater,order=order)[0]

        dO_av = dO_av[index]
        t_ = t[index]
        dO_st = dO_st[index]

        index = np.exp(dO_av) < max_dO
        dO_av = dO_av[index]
        t_ = t_[index]
        dO_st = dO_st[index]


        if len(dO_av) > 2 :
            # ax[idx].plot(t[index],dO_av[index])
   
            [m , q] , C = np.polyfit(t_, dO_av, 1, w = 1/dO_st,cov=True)

            
            # fig, ax = plt.subplots(1,1)
            # ax.plot(t[1:], q + t[1:]*m,linestyle='--')
            # ax.plot(t_, dO_av)
            # plt.show()

            [_ , q] , residual_exp , _ , _ , _ = np.polyfit(t_, dO_av, 1,w=1/dO_st,full=True)
            [m_pol , _] , resitual_pol , _ , _ , _ = np.polyfit(np.log(t_), dO_av, 1,w=1/dO_st,full=True)
            print(f'error pol: {resitual_pol}')
            print(f'error exp: {residual_exp}')
            # time.sleep(1)
            if residual_exp < resitual_pol:
                lyapunov.append(m)
                dlyapunov.append(np.sqrt(C[0,0]))
            else:
                lyapunov.append(0)
                dlyapunov.append(0)


            
            
            # ax[idx].plot(t[1:], q + t[1:]*m,linestyle='--')

    lyapunov.append(0)
    dlyapunov.append(0)

    lyapunov = np.array(lyapunov)
    dlyapunov = np.array(dlyapunov)

    index_max = np.argmax(lyapunov)

    return [lyapunov[index_max],dlyapunov[index_max]], t , dO
