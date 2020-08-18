import pickle
import time
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from graph_data import PFGraphDataset, elem_to_id, class_to_id, class_labels
from train_end2end import PFNet7
# Class for object that stores multiplicity and time for evaluation or unpacks them
from mult_and_time import TimeMultPack,SliceTimeMult

#loading kwargs pickle
#kwargs_dict = torch.load("../../model_init_data/model_kwargs.pkl")
pickle_in = open("../../model_init_data/model_kwargs.pkl","rb")
kwargs_dict = pickle.load(pickle_in)
#Run on either cpu or gpus

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


number = 0 #keeping track of progress

#binned data
#Iterator for the bins
for a_bin in range(3,13): 
    #the PFGraphDataset will look for the processed dir inside the "bin_{}k" dirs 
    binned = ("/home/aadi/praktika/particleflow/test/data/bin_{}k".format(a_bin))
    bin_data = PFGraphDataset(binned)
    bin_length = len(bin_data)
    #warm up model as the first eval takes more time usually
    if number == 0:
        test_data = bin_data[0]
        test_data.to(device)
        model(test_data)
    

    pickling_list = [] #list that will be pickled
    for j in range(30): #number of repetitions
    
        rep_list = [] #i-th repetiton data container
        #The sample data for evaluation is following bootstrapping. In this instance if a bin is bigger/equal
        #than 1000, then only 1000 random samples are taken, where same samples might occur also.
        #If the bin is smaller than 1000 then sample amoun is same size as the bin itself. Here also same 
        #samples might occur multiple times.
        if bin_length >= 1000:
            sample_nrs = np.random.randint(bin_length - 1 , size = 1000)#random number with certain amount
        else:
            sample_nrs = np.random.randint(bin_length - 1, size = bin_length)
        #random samples are used from the list of sample_nrs
        for i in sample_nrs:
            number = number +1
            print(number)
            data = bin_data[i] #i-th data of the random samples is initialized
            input_matrix  = data.x #input matrix from data
            multiplicity = input_matrix.shape[0] #extracting multiplicity
            data.to(device)#using gpus the device must be assigned for an input
            
            #model stopwatch
            start_time = time.time()
            model(data)
            end_time = time.time() -start_time
        
         
            mult_time = TimeMultPack(multiplicity,end_time) #packing multiplicity and time using the TimeMultPack ob.
            rep_list.append(mult_time.spit_out()) #appending to repetition list
    
    
        pickling_list.append(rep_list)

    #Dumping the mult and time pickle files
    torch.save(pickling_list ,"/home/aadi/praktika/pickle_lists_final3/mult_time_{}k_bin.pkl".format(a_bin))


