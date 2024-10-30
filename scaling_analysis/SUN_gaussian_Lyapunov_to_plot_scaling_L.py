# Simulate and save number_processor nearly sampled initial conditions
# The initial state is a N-mode Schrodinger cat state specified by parameter so that:
# |psi_cat> ~ |\gamma_1> + |\gamma_2> 
# Goal: compute the Lyapunov exponent as a functin of inhomongeneities in the local fields W and observe that as the 
# number of sites L changes the results do not qualitatively change (chaos is not depleted, thus it is not a finite size effect)

import sys
import multiprocessing
from scipy.signal import argrelextrema
sys.path.append(f'./library')
import chaos
import numpy as np
import matplotlib.pyplot as plt
import os


symbol = ['v','o','^','D']
style = ['-','--','dotted']
color = ['tab:blue','tab:red']
def cm2inch(value):
    return value/2.54

plt.rc('xtick' , labelsize=11)    # fontsize of the tick labels
plt.rc('ytick' , labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize =10)    # legend fontsize
plt.rc('axes'  , titlesize=11)     # fontsize of the axes set_title

plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })



text = ["IV${{}}^\star$","IV","IV${{}}^\star$","I"]

number_processors = multiprocessing.cpu_count()
do_plots = True
# number_processors = 1
main_dir = 'lyapunov_gaussian_states_swipe_W/'
json_file = ""
save_data = False
name_swipe = "W"
L_list = [5000,10000]
L_list = [10000,15000]

d = 3

W_list = [j * 0.01 for j in range(170)]
# W_list += [j * 0.1 for j in range(1,10)]
# W_list += [1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60]

W_plot = [0.03,0.2,1.1,1.6]
W_plot = [0.03,0.2,1.12,1.6]

fig, ax = plt.subplots(2,2,sharey=True,sharex=True,figsize=(cm2inch(9.6), cm2inch(4.4)))
# ax = ax.flatten()
# for a in ax[0]:
    # a.set_title('$\overline{\Sigma_{{1,2}}(t)}$')
ax[0][0].set_title('$\overline{\Sigma_{{1,2}}(t)}$',x=1)
# fig.suptitle('$\overline{\Sigma_{{1,2}}(t)}$')
for a in ax[1]:
    a.set_xlabel('$t\chi$')

ax = ax.flatten()

fig_l, ax_l = plt.subplots(1,1,figsize=(cm2inch(11.6), cm2inch(7.4)))

for index_L , L in enumerate(L_list):
    lyapunov_exponent = []
    dlyapunov_exponent = []
    W_exist = []



    for W in W_list:

        O = {}
        for i in range(d):
            for j in range(i+1,d):
                O[f's{i}{j}'] = []


        for idx in range(12):
            if W == 0:
                folder = f'{main_dir}SUN_Gaussian_L1_W{W:.3f}_index{idx}'
            else:
                folder = f'{main_dir}SUN_Gaussian_L{L}_W{W:.3f}_index{idx}'

            if os.path.isdir(folder):
                if idx == 0:
                    print(folder)

                for name in list(O.keys()):
                    if os.path.isfile(f'{folder}/{name}.npy'):
                        O[name].append(np.load(f'{folder}/{name}.npy'))
                time = np.load(f'{folder}/timesteps.npy')

                if W in W_plot and idx < 2:
                    # print(W)
                    ax[W_plot.index(W)].plot(time,O['s01'][-1]/O['s01'][-1][0],linewidth=0.8,linestyle=style[index_L],color=color[idx],label={L})
                    # ax[W_plot.index(W)].set_title(f'{W:.2f}')
                    x_text = 0.8
                    y_text = 0.3
                    # ax[W_plot.index(W)].text(x_text,y_text, f'$W/\chi={W:.2f}$ ({text[W_plot.index(W)]})', transform=ax[W_plot.index(W)].transAxes, size=6)
                    ax[W_plot.index(W)].text(x_text,y_text, f'{text[W_plot.index(W)]}', transform=ax[W_plot.index(W)].transAxes, size=8)

                
                exist = True
            else:
                exist = False

        if exist:
            W_exist.append(W)
            l , t,  dO = chaos.compute_lyapunov_exponent(O,time,max_dO=1E-3,order=1)
            lyapunov_exponent.append(l[0])
            dlyapunov_exponent.append(l[1])
            



            if do_plots and W in W_plot:
                fig_2 , ax_2 = plt.subplots(1,len(list(dO.keys())),figsize=(8,3))
                fig_2.suptitle(f'{W:.3f}')
                for name in list(dO.keys()):
                    dO[name] = np.array(dO[name])

                for index, name in enumerate(list(dO.keys())):
                    dO_av = np.exp(np.average(dO[name],axis=0))

                    k = argrelextrema(dO_av, np.greater,order=1)[0]
                    # j = np.where(dO_av < 1E-2)
                    # k = np.logical_and(k,j)
                    ax_2[index].plot(t[1:],np.array(dO_av))        

                    dO_av = dO_av[k]
                    t_ = t[k]

                    k = dO_av < 1E-3
                    dO_av = dO_av[k]
                    t_ = t_[k]


                    ax_2[index].plot(t_,np.array(dO_av))
                    ax_2[index].set_title(f'lyapu {name}')
                for a in ax_2:
                    a.set_yscale('log')
                plt.tight_layout()

    ax_l.errorbar(W_exist,lyapunov_exponent,yerr=dlyapunov_exponent,linestyle=style[index_L],color='black',linewidth=0.8,marker='.',markersize=2,label={L})

for a in ax:
    a.legend()

ax_l.legend()

    # for idx, item in enumerate(lyapunov_exponent):
    #     if item <= 0.004:
    #         lyapunov_exponent[idx] = 0



# for idx, w in enumerate(W_plot):
#     ax_l.plot(w,lyapunov_exponent[W_exist.index(w)],marker=symbol[idx],markersize=8,markeredgewidth=1,markeredgecolor='black')

ax_l.axhline(y=0, color='grey', linestyle='--',linewidth=0.8)
ax_l.set_xlabel('$W/\chi$')
ax_l.set_title('$\lambda$')
# fig_l.savefig(f"lyapunov_exponent_gaussian_state_L{L}_swipe_W.pdf",bbox_inches='tight',pad_inches = 0)

fig.subplots_adjust(wspace=0)
fig.subplots_adjust(hspace=0)

# fig.savefig(f"trajectories_lyapunov_exponent_gaussian_state_L{L}_swipe_W.pdf",bbox_inches='tight',pad_inches = 0)
plt.show()

