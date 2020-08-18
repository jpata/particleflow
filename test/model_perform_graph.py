import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch


from mult_and_time import TimeMultPack,SliceTimeMult



mean_eval_times=[]
bin_errors= []

evaluation = 0#counter
for a_bin in range(3,13):
    #loading pickle
    pickling_list = torch.load("/home/aadi/praktika/pickle_lists_final3/mult_time_{}k_bin.pkl".format(a_bin))    
    #30 repetitions individual mean will be stored in repetitions
    repetitions = []
    #a rep is taken from 30 reps
    for rep in pickling_list:
            
        #Single repetition mean is tored in a_rep_mean
        a_rep_mean =[]
        #A TimeMultPack object is taken
        for mult_time in rep:
            #exctraction of evaluation time
            time = SliceTimeMult(mult_time)
            evaluation = evaluation +1
            print(evaluation)
            
            a_rep_mean.append(time.get_time())    
        #mean is calculated for single repetition     
        a_rep_mean = np.mean(a_rep_mean)
        repetitions.append(a_rep_mean)    

    

    #mean of all 30 repetition
    bin_mean = np.mean(repetitions)
    bin_mean.tolist()
    
    mean_eval_times.append(bin_mean)
    #standard deviation is calculated for 30 reps
    bin_std =np.std(repetitions)
    bin_std.tolist()

    bin_errors.append(bin_std)
    



#Plotting
fig, ay = plt.subplots()
ay.set_title('Event multiplicity effect on model evaluation time')
ay.set_xlabel('Multiplicity bins')
ay.set_ylabel('Evaluation time for a single event')







x_value = [3,4,5,6,7,8,9,10,11,12]
x_labels = [str(i)+'k' for i in range(3,13)]
error_mark= bin_errors
y_values = mean_eval_times

ay.bar(x_value, mean_eval_times,capsize=7,yerr= error_mark,align='center',color='blue', alpha = 0.6,ecolor='red')
ay.set_xticks(x_value)
ay.set_xticklabels(x_labels)
#Saving/showing
#plt.show()
fig.savefig('Fifth_graph.png') 


