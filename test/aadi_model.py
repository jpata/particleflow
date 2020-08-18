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



#loading predetermined weights
# ON CPU
#device = torch.device('cpu')
#predet_weights = torch.load("../../model_init_data/weights.pth",map_location = device)

# ON GPU
device = torch.device("cuda")
predet_weights = torch.load("../../model_init_data/weights.pth")


#loading model

model = PFNet7(**kwargs_dict)
missing_key_fix = {k.replace("module.", ""): v for k, v in predet_weights.items()}
model.load_state_dict(missing_key_fix)
#### on GPU
model.to(device)
####
model.eval()

#Bootstrapping 
#sample_nrs = np.random.randint(13601 , size = 1000)#random number with certain amount

number = 0 #keeping track of progress

#binned data
for a_bin in range(3,13): 
    binned = ("/home/aadi/praktika/binned_data/bin_{}k".format(a_bin))       #processed
    bin_data = PFGraphDataset(binned)
    bin_length = len(bin_data)
    #warm up model
    if number == 0:
        test_data = bin_data[0]
        test_data.to(device)
        model(test_data)
    

    pickling_list = [] #list that will be pickled
    for j in range(30): #number of repetitions
    
        rep_list = [] #i-th repetiton data container
        if bin_length >= 1000:
            sample_nrs = np.random.randint(bin_length - 1 , size = 1000)#random number with certain amount
        else:
            sample_nrs = np.random.randint(bin_length - 1, size = bin_length)
    
        for i in sample_nrs:
            number = number +1
            print(number)
            data = bin_data[i] #i-th data
            input_matrix  = data.x #input matrix from data
            multiplicity = input_matrix.shape[0] #multiplicity
            data.to(device)
            
            #model stopwatch
            start_time = time.time()
            model(data)
            end_time = time.time() -start_time
        
         
            mult_time = TimeMultPack(multiplicity,end_time) #packing multiplicity and time 
            rep_list.append(mult_time.spit_out()) #appending to repetition list
    
    #
        pickling_list.append(rep_list)

    #print(pickling_list)
    pickle_out = open("/home/aadi/praktika/pickle_lists_final2/mult_time_{}k_bin.pkl".format(a_bin),"wb")
    pickle.dump(pickling_list, pickle_out)
    pickle_out.close()


