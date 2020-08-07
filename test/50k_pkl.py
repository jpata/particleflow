#!/use/bin/env python3
import torch
import torch_geometric
import pickle
import numpy as np

#Turning data in to a pickle of 50000


from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from graph_data import PFGraphDataset, elem_to_id, class_to_id, class_labels

dataset_path = "/home/joosep/particleflow/data/TTbar_14TeV_TuneCUETP8M1_cfi"

#Goal: measure the evaluation cost of the MLPF model as a function of input multiplicity
if __name__ == "__main__":
    full_dataset = PFGraphDataset(dataset_path)
    i = 0
    input_multiplicity =[]
    #bunches of 10
    for data_items in full_dataset:
        #import pdb
        #pdb.set_trace()
        #print(data_items[9])

        #loop over each event in the bunch
        for data in data_items:
            i = i +1
            print(i)
            #print(data)

            #get the input matrix
            #input_matrix = data
            #print(input_matrix)
            #print("input_matrix.shape=", input_matrix.shape)

            #this is the number of input elements in the event
            input_multiplicity.append(data)
            #print(input_multiplicity)
            if i == 50000:
                #print(input_multiplicity[0])

                pickle_out = open("50k_pkl.pkl","wb")
                pickle.dump(input_multiplicity,pickle_out)
                pickle_out.close()
                quit()

            #task 1: plot the distribution of the input multiplicities across the events using numpy.histogram and matplotlib.histogram

            #task 2: save the `data` object using torch.save(data, "data/TTbar_14TeV_TuneCUETP8M1_cfi/bin_i/file_j.pt") to
            #subfolders based on the input multiplicity binning


