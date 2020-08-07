#!/use/bin/env python3

#on manivald: singularity exec -B /home /home/software/singularity/base.simg:latest python3 test/evaluate_timing.py


import torch
import torch_geometric
import pickle
import numpy as np




from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from graph_data import PFGraphDataset, elem_to_id, class_to_id, class_labels

dataset_path = "/home/joosep/particleflow/data/TTbar_14TeV_TuneCUETP8M1_cfi"
dataset_path_aadi = "/home/aadi/praktika/particleflow/test/data/"
#Goal: measure the evaluation cost of the MLPF model as a function of input multiplicity
if __name__ == "__main__":
    full_dataset = PFGraphDataset(dataset_path_aadi)
    count= 0
    mistakes=[]

    #range check
    for data in full_dataset:
        input_matrix = data.x
        #this is the number of input elements in the event
        input_multiplicity = input_matrix.shape[0]
        print(count)
        count = count + 1
        

        if 4000 <= input_multiplicity < 5000:
            print(f'{input_multiplicity} OK')
        else:
            print(f'{input_multiplicity} NOT OK')
            mistakes.append(input_multiplicity)
        if count == 465:
            print(mistakes)
            quit()  






















































