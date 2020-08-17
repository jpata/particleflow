import pickle
import time
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from graph_data import PFGraphDataset, elem_to_id, class_to_id, class_labels

from train_end2end import PFNet7


from mult_and_time import TimeMultPack,SliceTimeMult

#loading kwargs pickle
pickle_in = open("../../model_init_data/model_kwargs.pkl","rb")
kwargs_dict = pickle.load(pickle_in)

#binned data 
binned = "/home/aadi/praktika/binned_data/"#processed

#loading predetermined weights
device = torch.device('cpu')
predet_weights = torch.load("../../model_init_data/weights.pth",map_location = device)
#6k_bin at the moment
bin_data = PFGraphDataset(binned)
print(len(bin_data))


#loading model
model = PFNet7(**kwargs_dict)
missing_key_fix = {k.replace("module.", ""): v for k, v in predet_weights.items()}
model.load_state_dict(missing_key_fix)
model.eval()

#Bootstrapping 
#sample_nrs = np.random.randint(13601 , size = 1000)#random number with certain amount

number = 0
pickling_list = [] #list that will be pickled
for j in range(10): #number of repetitions
    
    rep_list = [] #i-th repetiton data container
    sample_nrs = np.random.randint(5 , size = 6)#random number with certain amount
    
    for i in sample_nrs:
        number = number +1
        print(number)
        data = bin_data[i] #i-th data
        input_matrix  = data.x #input matrix from data
        multiplicity = input_matrix.shape[0] #multiplicity

        #model stopwatch
        start_time = time.time()
        model(data)
        end_time = time.time() -start_time
        
         
        mult_time = TimeMultPack(multiplicity,end_time) #packing multiplicity and time 
        rep_list.append(mult_time.spit_out()) #appending to repetition list
    
    #
    pickling_list.append(rep_list)

#print(pickling_list)
pickle_out = open("/home/aadi/praktika/pickle_lists/mult_time_12k_bin.pkl","wb")
pickle.dump(pickling_list, pickle_out)
pickle_out.close()


