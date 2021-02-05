import pickle
import argparse
import time
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from graph_data import PFGraphDataset, elem_to_id, class_to_id, class_labels
from train_end2end import PFNet7
# Class for object that stores multiplicity and time for evaluation or unpacks them
from mult_and_time import TimeMultPack,SliceTimeMult

def LoadModelCPU(kwargs_path,weights_path):
    try:
        kwargs_dict = torch.load(kwargs_path)
    except RuntimeError:
        pickle_in = open(kwargs_path,"rb")
        kwargs_dict = pickle.load(pickle_in)

    device = torch.device('cpu')
    predet_weights = torch.load(weights_path,map_location = device)
    

    model = PFNet7(**kwargs_dict)
    missing_key_fix = {k.replace("module.", ""): v for k, v in predet_weights.items()}
    model.load_state_dict(missing_key_fix)
    model.eval()
     
    return model, device

def LoadModelGPU(kwargs_path, weights_path):
    try:
        kwargs_dict = torch.load(kwargs_path)
    except RuntimeError:
        pickle_in = open(kwargs_path,"rb")
        kwargs_dict = pickle.load(pickle_in)
    device = torch.device("cuda")
    predet_weights = torch.load(weights_path)
    
    model = PFNet7(**kwargs_dict)
    missing_key_fix = {k.replace("module.", ""): v for k, v in predet_weights.items()}
    model.load_state_dict(missing_key_fix)
    model.to(device)
    model.eval()
    
    return model, device

def BinIterator(a_bin, bins_dir_path):
        #the PFGraphDataset will look for the processed dir inside the "bin_{}k" dirs 
    binned = (bins_dir_path+"/bin_{}k".format(a_bin))
    bin_data = PFGraphDataset(binned)
    bin_length = len(bin_data)
    return  bin_data, bin_length


def BootStrap(bin_length,sample_size):
    #The sample data for evaluation is following bootstrapping. In this instance if a bin is bigger/equal
    #than 1000, then only 1000 random samples are taken, where same samples might occur also.
    #If the bin is smaller than 1000 then sample amoun is same size as the bin itself. Here also same 
    #samples might occur multiple times.
    if bin_length >= sample_size:
        sample_nrs = np.random.randint(bin_length - 1 , size = sample_size)#random number with certain amount
    else:
        sample_nrs = np.random.randint(bin_length - 1, size = bin_length)
    
    return sample_nrs

def StopWatchCPU(bin_data,sample_nrs,file_nr, model):
    data = bin_data[file_nr] #i-th data of the random samples is initialized
    input_matrix  = data.x #input matrix from data
    multiplicity = input_matrix.shape[0] #extracting multiplicity

    #model stopwatch
    start_time = time.time()
    model(data)
    end_time = time.time() -start_time


    return multiplicity,end_time



def StopWatchGPU(bin_data,sample_nrs,file_nr,model,device):
    data = bin_data[file_nr] #i-th data of the random samples is initialized
    input_matrix  = data.x #input matrix from data
    multiplicity = input_matrix.shape[0] #extracting multiplicity
    data.to(device)#using gpus the device must be assigned for an input

    #model stopwatch
    start_time = time.time()
    model(data)
    end_time = time.time() -start_time


    return multiplicity,end_time
    

def MainCPU(kwargs_path,weights_path,bins_dir_path,save_path, repetitions,sample_size):
    model, device = LoadModelCPU(kwargs_path,weights_path)
    progress = 0
    
    for a_bin in range(3,13):
        bin_data, bin_length = BinIterator(a_bin,bins_dir_path)
        #warm up model as the first eval takes more time usually
        if progress == 0:
            test_data = bin_data[0]
            model(test_data)

        pickling_list=[]#list that will be pickled
    
        for j in range(repetitions):#number of repetitions
            rep_list = []#i-th repetition data container
            sample_nrs = BootStrap(bin_length,sample_size) 
            
            for file_nr in sample_nrs:
                progress = progress +1
                print(progress)
                multiplicity, end_time =  StopWatchCPU(bin_data,sample_nrs,file_nr, model)


                mult_time = TimeMultPack(multiplicity,end_time) #packing multiplicity and time using the TimeMultPack ob.
                rep_list.append(mult_time.spit_out()) #appending to repetition list


            pickling_list.append(rep_list)

        #Dumping the mult and time pickle files
        torch.save(pickling_list ,save_path+"/mult_time_{}k_bin.pkl".format(a_bin))
    


def MainGPU( kwargs_path, weights_path, bins_dir_path, save_path, repetitions, sample_size):
    model, device = LoadModelGPU(kwargs_path,weights_path)
    progress = 0

    for a_bin in range(3,13):
        bin_data, bin_length = BinIterator(a_bin,bins_dir_path)
        #warm up model as the first eval takes more time usually
        if progress == 0:
            test_data = bin_data[0]
            test_data.to(device)
            model(test_data)

        pickling_list=[]#list that will be pickled

        for j in range(repetitions):#number of repetitions
            rep_list = []#i-th repetition data container
            sample_nrs = BootStrap(bin_length,sample_size)

            for file_nr in sample_nrs:
                progress = progress +1
                print(progress)
                multiplicity, end_time =  StopWatchGPU(bin_data,sample_nrs,file_nr, model,device)


                mult_time = TimeMultPack(multiplicity,end_time) #packing multiplicity and time using the TimeMultPack ob.
                rep_list.append(mult_time.spit_out()) #appending to repetition list


            pickling_list.append(rep_list)

        #Dumping the mult and time pickle files
        torch.save(pickling_list ,save_path+"/mult_time_{}k_bin.pkl".format(a_bin))



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--kwargs_path",help="Path were the kwargs file is located as a pkl file for the model ",type = str )
    parser.add_argument("--weights_path", help="Path where the weights.pth file is located for the model ", type =str)
    parser.add_argument("--bins_dir_path",help ="Path where the bin_{}k files are located",type =str)
    parser.add_argument("--save_path", help ="Path where the pickled files on model time evaluation are stored", type =str)
    parser.add_argument("--repetitions", help = "The amount of repetitions of the evaluations ",default = 30,  type = int)
    parser.add_argument("--sample_size", help ="The amount of maximum random samples for bootstrap function",default = 1000, type = int)
    parser.add_argument("--device", help = "The device where the program will be ran on. Either cpu or gpu",type =str)

    args = parser.parse_args()

    if args.device == "cpu":
        print("Running on cpu")
        MainCPU(args.kwargs_path,args.weights_path,args.bins_dir_path,args.save_path,args.repetitions,args.sample_size)
    if args.device == "gpu":
        MainGPU(args.kwargs_path,args.weights_path,args.bins_dir_path,args.save_path,args.repetitions,args.sample_size)

