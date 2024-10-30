import functools
import os
import pandas as pd
import csv
from pathlib import Path
import numpy as np
import json

def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)


def check_existence_data(folder,args,name_csv='data'):

    list_args   = list(args.keys())

    with open(f'{folder}/{name_csv}.csv') as data_csv:
        df = pd.read_csv(data_csv)

        header = df.columns.values.tolist()
        index  = df.index

        if list_args.sort() != header.sort():
            print('Headers do not match')
            return -1

        c = []
        
        for name in args:
            c.append(df[name]==args[name])                
        condition = conjunction(*c)

        if len(index[condition].to_list()) > 0:
            print('Data already exist')
            return 1

        else:
            return 0



def get_index_csv(folder,args,name_csv='data'):
    list_args = list(args.keys())
    with open(f'{folder}/{name_csv}.csv') as data_csv:
        df = pd.read_csv(data_csv)

        header = df.columns.values.tolist()
        index  = df.index

        print(index)

        if list_args.sort() != header.sort():
            print('Headers do not match')
            return -1

        c = []
        
        for name in args:
            c.append(df[name]==args[name])                
        condition = conjunction(*c)



        if len(index[condition].to_list()) > 0:
            print('Data already exist')
            return index[condition].to_list()[-1]
        else:
            return len(index.to_list()) + 1


def get_last_index(folder,args,name_csv='data'):
    list_args = list(args.keys())
    with open(f'{folder}/{name_csv}.csv') as data_csv:
        df = pd.read_csv(data_csv)
        index  = df.index

        
        return len(index.to_list()) + 1



def create_directory_and_csv(main_dir,name_folder,header):

    folder      = f"{main_dir}{name_folder}"
    if os.path.isdir(folder)==False:
        Path(folder).mkdir()

    with open(f'{folder}/data.csv', mode='a+') as data_csv:
        data = csv.writer(data_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if os.stat(data_csv.name).st_size == 0:
            data.writerow(header)

    return folder , True



def save_data(folder,name_csv,prefix,index,d,args_csv, args_json, obs,times):


    with open(f'{folder}/{name_csv}.csv', mode='a+') as data_csv:
        data = csv.writer(data_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data.writerow(list(args_csv.values()))

    save_folder = f"{folder}/{prefix}{d}_{index}"

    if os.path.isdir(save_folder)==False:
        Path(save_folder).mkdir()

    for _ , O in enumerate(obs):
        np.save(f'{save_folder}/{O}.npy',obs[O])
    np.save(f'{save_folder}/time.npy',times)

    with open(f'{save_folder}/input.json','w') as f:
        json.dump(args_json,f)


    return save_folder, True