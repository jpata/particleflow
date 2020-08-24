#!/use/bin/env python3
import argparse
import torch
import torch_geometric
import pickle
import numpy as np
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from graph_data import PFGraphDataset, elem_to_id, class_to_id, class_labels

#This program makes the data, which will be used for evaluation into a pickle file.
#Then it is ready for sorting into bins of different multiplicity.



#First set the path of the data.
#dataset_path = "/home/joosep/particleflow/data/TTbar_14TeV_TuneCUETP8M1_cfi"

def Main(path_in,path_out,file_name,bunch_size):
    full_dataset = PFGraphDataset(path_in)
    i = 0 #counter
    list_for_pickle =[] #The list that will be pickled


    #This dataset is in bunches of ten. The output will be singular for each event
    for data_items in full_dataset:
        

        #loop over each event in the bunch
        for data in data_items:
            i = i +1
            print(i)

            list_for_pickle.append(data)
            ##NB!! The PFGraphDataset will be looking endlessly for its datatype.
            ##Make sure you limit it. 
            if i == len(full_dataset)*bunch_size:
                #print(input_multiplicity[0])
                torch.save(list_for_pickle, path_out +"/"+file_name)
                print("Pkl file successfully created.")
                quit()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bunch_size",help="How many items in a bunch. If none then default is 1",type=int,default=1)
    parser.add_argument("--path_in", help="The input path of the data_{}.pt files.", type=str)
    parser.add_argument("--path_out", help="The output file location of the dataset in .pkl file",type=str)
    parser.add_argument("--file_name",help="The name of the .pkl file",type= str)
    
    args = parser.parse_args()

    program_run =Main(args.path_in,args.path_out,args.file_name,args.bunch_size)



