#!/use/bin/env python3
import torch
import torch_geometric
import pickle
import numpy as np
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from graph_data import PFGraphDataset, elem_to_id, class_to_id, class_labels

#This program makes the data, which will be used for evaluation into a pickle file.
#Then it is ready for sorting into bins of different multiplicity.


#First set the path of the data.
dataset_path = "/home/joosep/particleflow/data/TTbar_14TeV_TuneCUETP8M1_cfi"


if __name__ == "__main__":
    full_dataset = PFGraphDataset(dataset_path)
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
            if i == 50000:
                #print(input_multiplicity[0])
                torch.save(list_for_pickle, '50k_data.pkl')
                quit()

