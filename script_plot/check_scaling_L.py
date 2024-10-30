L_list = [10000,20000]
D = 3
name_obs = 's02'
style = ['-','--']

plot_dir = '/home/ricval/Documenti/Cavity/DPT_SUn_paper/final_images'
prefix_dir = '/home/ricval/Documenti/Cavity_python/data_paper/'

main_dir_list =  [f'{prefix_dir}SU3_Gaussian_p_vs_W/', f'{prefix_dir}SU3_Gaussian_swipe_p_and_W_noise1E-5/']

height_images = 8
width_images = 8.6 * 2 / 2.3

############################################################

p_list = [0.175,0.32]
p_list = [0.20,0.32]

W_list = [[0.01,0.1],[0.08,0.18]]

text_list = [["IV","III"],["IV${{}}^\star$","IV"]]


for idx_p , p in enumerate(p_list):

    text = text_list[idx_p]

    L = L_list[idx_p]
    main_dir = main_dir_list[idx_p]
    fig, (ax_1,ax_2,ax_3,ax_4) = plt.subplots(2,2,sharex=False,sharey=False,figsize=(cm2inch(width_images), cm2inch(height_images)),constrained_layout=True)

    ax_1.get_xaxis().set_visible(False)
    ax_2.get_xaxis().set_visible(False)
    ax_2.get_yaxis().set_visible(False)
    ax_4.get_yaxis().set_visible(False)

    ax_1.set_title('$\Sigma_{{1,3}}$',x=1,fontsize=8)

    for a in [ax_3]:   
        a.set_xlabel('$t\chi\mathcal{{N}}_a$',fontsize=8,x=1)

    ax_inset = [ax_1,ax_2,ax_3,ax_4]
    for a in ax_inset:
        a.tick_params(axis='both', which='major', labelsize=8)
        a.tick_params(axis='both', which='minor', labelsize=8)

    for a in [ax_1,ax_3]:
        a.spines['right'].set_visible(False)
    for a in [ax_2,ax_4]:
        a.spines['left'].set_visible(False)
        a.get_yaxis().set_visible(False)

        
    
    for index_W, W in enumerate(W_list[idx_p]):
        print(W)
        O = {}
        for i in range(D):
            for j in range(i+1,D):
                O[f's{i}{j}'] = []

        for idx in range(3):
            
            folder = f'{main_dir}SUN_Gaussian_L{L}_p{p:.3f}_W{W:.3f}_index{idx}'
            print(folder)

            if os.path.isdir(folder):
                

                for name in list(O.keys()):
                    if os.path.isfile(f'{folder}/{name}.npy'):
                        O[name] = np.abs(np.load(f'{folder}/{name}.npy'))
                time = np.load(f'{folder}/timesteps.npy')

                start = int(len(time)/3)
                t = time[start:]
                obs = O[name_obs][start:]
                dt = t[1]-t[0]
                N  = len(t)
                yf = fft(obs)
                xf = fftfreq(N, dt)*2*math.pi
                # ax_1.plot(xf , 2.0/N * np.abs(yf), linestyle='-',linewidth=linewidth , alpha=1)


                for index_O, name in enumerate(list(O.keys())):
                    if name == name_obs:
                        index_O = 0
             
                        ax_inset[index_W*2].plot(time,np.abs(O[name][-1]),linewidth=0.8,linestyle=style[idx],color=color[idx])
                        ax_inset[index_W*2+1].plot(time,np.abs(O[name][-1]),linewidth=0.8,linestyle=style[idx],color=color[idx])

                        x_text = 0.63
                        y_text = 0.68
                        ax_inset[index_W*2+1].text(x_text,y_text, f'{text[index_W]}', transform=ax_inset[index_W*2+1].transAxes,  bbox=props, size=font_size)

                    # else:
                    #     ax1.plot(time,np.abs(O[name]), linewidth=linewidth , linestyle=linestyle[index_O],alpha=1)                
                    #     ax2.plot(time,np.abs(O[name]), linewidth=linewidth , linestyle=linestyle[index_O],alpha=1)                

           
    ax1.set_xlim([0,150])
    # ax2.set_xlim([2000,2300])

    ax2.set_xlim([time[-1]-75,time[-1]])

    # ax1.xaxis.set_ticks(np.arange(0, 150, 50 ))
    # ax2.xaxis.set_ticks(np.arange(2900, 3010, 50 ))
    ax1.legend(ncol=2,frameon=False,loc='lower center',bbox_to_anchor=(1, -0.6),handlelength=1)

    fig.supxlabel('$t\chi\mathcal{{N}}_a$',y=0.15,x=0.53)
    fig.suptitle(f'$p={p:.3f}$',y=0.93,x=0.53)


    plt.tight_layout()

    fig.subplots_adjust(wspace=0.06)
    fig.subplots_adjust(bottom=0.3)


    d = .01  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax_1.transAxes, color='k', clip_on=False)
    ax_1.plot((1-d, 1+d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax_1.plot((1-d, 1+d), (- d, + d), **kwargs)  # bottom-left diagonal

    kwargs.update(transform=ax_2.transAxes)  # switch to the bottom axes
    ax_2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax_2.plot((- d, + d), (- d, + d), **kwargs)  # bottom-right diagonal

    kwargs = dict(transform=ax_3.transAxes, color='k', clip_on=False)
    ax_3.plot((1-d, 1+d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax_3.plot((1-d, 1+d), (- d, + d), **kwargs)  # bottom-left diagonal

    kwargs.update(transform=ax_4.transAxes)  # switch to the bottom axes
    ax_4.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax_4.plot((- d, + d), (- d, + d), **kwargs)  # bottom-right diagonal

    
    plt.savefig(f'{plot_dir}/dynamics_and_fourier_Gaussian_SU3_swipe_W_p{p:.4f}_no_spectrum.pdf',bbox_inches='tight',pad_inches = 0)