import pickle
import time
import numpy as np
import matplotlib.pyplot as plt


from mult_and_time import TimeMultPack,SliceTimeMult



mean_eval_times=[]
bin_errors= []

evaluation = 0
for a_bin in range(3,13):
    #loading pickle
    pickle_in = open("/home/aadi/praktika/pickle_lists_final2/mult_time_{}k_bin.pkl".format(a_bin),"rb")
    pickling_list = pickle.load(pickle_in)    
    
    repetitions = []
    for rep in pickling_list:
            

        a_rep_mean =[]
        for mult_time in rep:
            time = SliceTimeMult(mult_time)
            evaluation = evaluation +1
            print(evaluation)
            a_rep_mean.append(time.get_time())    

        a_rep_mean = np.mean(a_rep_mean)
        repetitions.append(a_rep_mean)    

    


    bin_mean = np.mean(repetitions)
    bin_mean.tolist()

    mean_eval_times.append(bin_mean)

    bin_std =np.std(repetitions)
    bin_std.tolist()

    bin_errors.append(bin_std)
    




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

#plt.show()
fig.savefig('Fourth_graph.png') 


