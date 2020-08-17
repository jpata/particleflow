
#Multiplicities = [123,234,122,567,456,789,876,534]
#running_time = [1.2,1.2,1.5,1.3,1.1,1.2,1.7,1.1]
#results=[]

class TimeMultPack:
    
    
    def __init__(self,multiplicity,time):
        self.multiplicity = multiplicity
        self.time = time

    def spit_out(self):
        result=[]
        result.append(self.multiplicity)
        result.append(self.time)
        
        return result




class SliceTimeMult:


    def __init__(self,a_list):
        self.a_list = a_list


    def get_time(self):
        a_time = self.a_list[1]
        return a_time
    

    def get_multiplicity(self):
        a_multiplicity = self.a_list[0]
        return  a_multiplicity



#lass PackReps:
#   def __init__(self,a_list):
#   self.a_list = a_list


#Multiplicities = [123,234,122,567,456,789,876,534]
#running_time = [1.2,1.2,1.5,1.3,1.1,1.2,1.7,1.1]
#results=[]

#for i in range(8):
#    time_mult= TimeMultPack(Multiplicities[i],running_time[i])
#    results.append(time_mult.spit_out())
#print(results)

#time = SliceTimeMult(results[2])
#print(time.get_multiplicity())
#print(time.get_time())

