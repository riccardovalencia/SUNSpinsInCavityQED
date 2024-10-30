
import numpy as np
import pandas as pd
import os,  math, functools
from scipy.fft import fft, fftfreq



def get_obs(directory, name_obs, file_extension):
    name_file_obs = '{}/{}.{}'.format(directory,name_obs,file_extension)
    if os.path.isfile(name_file_obs) and file_extension == 'npy':
        obs = np.load(name_file_obs)
        return np.array(obs), True
    else:
        return 0, False
    

# time average of an observable
                                     
def time_average( observable , time):
    observable_average = []
    dt = []
    for i in range(len(time)-1):
        dt.append(time[i+1]-time[i])

    obs_av_single_dt = 0
    for i in range(len(dt)):
        obs_av_single_dt += (observable[i+1] + observable[i])/2 * dt[i] 
        observable_average.append(obs_av_single_dt / (time[i+1]-time[0]) )
       
    observable_average = np.array(observable_average)
    return observable_average


def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)


# return the dataset which satisfies the conditions imposed.

def get_subdataset(df,data_filter_list):
    name_filter   = [data_filter[0] for data_filter in data_filter_list]
    values_filter = [data_filter[1] for data_filter in data_filter_list]
    
    header = df.columns.values.tolist()
    index  = df.index
    c = []
    for name in name_filter:
        if name in df.head():
            c.append(df[name] == values_filter[name_filter.index(name)])

    condition = conjunction(*c)
    data_frame = df[condition]
#     print(data_frame)
    return data_frame
    

    
def get_directories(path,csv_file,param_fixed_list,param_switch_list):
   
    # select sub_dataset with some parameters
    
    with open('{}'.format(csv_file)) as data_csv:
        df      = pd.read_csv(data_csv)        

        # filter dataset based on fixed parameters
        df_filtered = get_subdataset(df,param_fixed_list)
        
        # sort filtered dataset based on the switch parameter
        # for j in range(len(param_switch_list)):
        #     df_filtered = df_filtered.sort_values(by=[param_switch[j][0]],ascending=True)

        # sort filtered dataset based on the switch parameter
        for param_switch in param_switch_list:
            df_filtered = df_filtered.sort_values(by=[param_switch[0]],ascending=True)

        switch_exist = []
        for j in range(len(param_switch_list)):
            switch_exist.append(df_filtered[param_switch_list[j][0]].tolist())
            
        indices = df_filtered.index.tolist()
        
        
        # get folder names containing the desired simulations
        list_directory = []
        for index in indices:
            list_directory += [filename for filename in os.listdir(path) if filename.endswith('_{}_processed'.format(index+1))]
        
        return list_directory, switch_exist
        
        
def get_obs_vs_time(path,csv_file,param_fixed_list,param_switch_list,name_obs, perform_average, compute_fluctuactions=False):
    
    directories, switch = get_directories(path,csv_file,param_fixed_list,param_switch_list)
    switch_obs = []
    obs_list = []
    time_list = []
    
    for directory in directories:
        obs, success = get_obs('{}/{}'.format(path,directory), name_obs, 'npy')
        timesteps, _ = get_obs('{}/{}'.format(path,directory), 'timesteps', 'npy')

        if success:
            
            if compute_fluctuactions:
                start = int(len(timesteps)*0.5)
#                 start = 1
#                 if obs[0]>1E-4:
#                     obs /= obs[0]
                obs_av = time_average(obs[start:], timesteps[start:])
                timesteps = timesteps[start+1:]

                obs = obs[start+1:] - obs_av[-1]#/(np.max(obs[start+1:]) + obs_av[-1])
#                 obs /= obs_av[-1]
                if perform_average:
                    obs = np.sqrt(time_average(np.square(obs), timesteps))
#                     obs = [0,np.sqrt(np.average(np.square(obs)))]
                    timesteps = timesteps[1:]


            elif perform_average:
                start = int(len(timesteps)*0.5)
                obs = time_average(obs[start:], timesteps[start:])
                timesteps = timesteps[start+1:]
                    
            for j in range(len(switch)):
                switch_obs.append(switch[j][directories.index(directory)])
            obs_list.append(obs)
            time_list.append(timesteps)
    
        else:
            return [float('nan')], [float('nan')], [float('nan')]
    return time_list, obs_list, switch_obs


def fourier_analysis(obs,time, threshold):
    
        dt_l = []
        for k in range(len(time)-1):
            dt_l.append(time[k+1]-time[k])
            
        if obs[0]>1E-6:
            obs /= obs[0]
                
        dt = np.average(dt_l)
        yf = fft(obs)
        N  = len(obs)
        xf = fftfreq(N, dt)[:N//2]*2*math.pi

        yf_abs = 2.0/N * np.abs(yf)
        indices = yf_abs > threshold
#         print(indices)
        yf_clean = indices * yf
        return int(len(np.where(yf_clean > 1E-6)[0])/2)