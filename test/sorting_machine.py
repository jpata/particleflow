#!/use/bin/env python3
import argparse
import os
import torch
import torch_geometric
import pickle
import numpy as np
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from graph_data import PFGraphDataset, elem_to_id, class_to_id, class_labels

#Sorting data into bindirectories according to input multiplicity of an event
#The bins are filled with data where input multiplicity is in ranges of 3k-4k, 4k-5k,..., 12k-13k



#create bin directories
#NB! The processed drectory is requierd by the PFGraphDataset
###
def create_bin_dirs(path_out):
    for path_nr in range(3,13):
        path = path_out+"/bins/bin_{}k/processed".format(path_nr)
        try:
            os.makedirs(path)
        except OSError:
            print("Creation of the bin directories to %s failed. Terminating Program!" % path)
            quit()
        else:
            print("Successfully created the bin directories to %s" % path)

###


def Main(pickle_file,path_out):
    data_items = torch.load(pickle_file)
    #File numbering init
    #NB! Here also the PFGraphDataset needs its own form of data presentation
    #The data must be named like "data_{}.pt" and it has to start from 0
    k3 = 0
    k4 = 0
    k5 = 0
    k6 = 0
    k7 = 0
    k8 = 0
    k9 = 0
    k10 = 0
    k11 = 0
    k12 = 0

    k = 0# The progress counter

    for j in range(0,len(data_items)):
        print(f'fail nr:{k}')
        #Taking the input matrix from data with extension ".x"
        input_matrix = data_items[j].x
        #i is input multiplicity
        i = input_matrix.shape[0]
        k = k +1
    
        #Bin sorting
        #If the multiplicity falls into a certain range bin, the program stores it there
        #bin 3k-4k
        if i in range(3000,4000):
            torch.save(data_items[j], path_out+"/bins/bin_3k/processed/data_{}.pt".format(k3))
            k3 = k3 +1

        #bin 4k-5k
        if i in range(4000,5000):
            torch.save(data_items[j], path_out+"/bins/bin_4k/processed/data_{}.pt".format(k4))
            k4= k4 +1

        #bin 5k-6k
        if i in range(5000,6000):
            torch.save(data_items[j], path_out+"/bins/bin_5k/processed/data_{}.pt".format(k5))
            k5 = k5 +1

        #bin 6k-7k
        if i in range(6000,7000):
            torch.save(data_items[j], path_out+"/bins/bin_6k/processed/data_{}.pt".format(k6))
            k6 = k6 + 1

        #bin 7k-8k
        if i in range(7000,8000):
            torch.save(data_items[j], path_out+"/bins/bin_7k/processed/data_{}.pt".format(k7))
            k7 = k7 + 1
        #bin 8k-9k
        if i in range(8000,9000):
            torch.save(data_items[j], path_out+"/bins/bin_8k/processed/data_{}.pt".format(k8))
            k8 = k8 + 1

        #bin 9k-10k
        if i in range(9000,10000):
            torch.save(data_items[j], path_out+"/bins/bin_9k/processed/data_{}.pt".format(k9))
            k9 = k9 + 1

        #bin 10k-11k
        if i in range(10000,11000):
            torch.save(data_items[j], path_out+"/bins/bin_10k/processed/data_{}.pt".format(k10))
            k10 = k10 + 1

        #bin 11k-12k
        if i in range(11000,12000):
            torch.save(data_items[j], path_out+"/bins/bin_11k/processed/data_{}.pt".format(k11))
            k11 = k11 + 1

        #bin 12k-13k
        if i in range(12000,13000):
            torch.save(data_items[j], path_out+"/bins/bin_12k/processed/data_{}.pt".format(k12))
            k12 = k12 + 1        

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_file",help="Path to the pickle file for sorting",type=str)
    parser.add_argument("--bins_path",help=" Path were files will be sorted in form as .../bins/bin_{j}k/data_{i}.pt",type = str)

    args = parser.parse_args()
    create_dirs = create_bin_dirs(args.bins_path) 
    program_run =Main(args.pickle_file,args.bins_path)





                                                                    
