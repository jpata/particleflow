

class TimeMultPack:
    '''
    TimeMultPack packs together multiplicity and evaluation time by the model for the same event.
    
    Paramaters
    ---------

    multiplicity : int
        This resembles the event multiplicity

    time : int/float
        This resembles the time a each event takes with the model evaluation

    Attributes
    ----------

    multiplicity : int
        This is where the multiplicity is stored
    
    time : int/float
        This is where time of the model evaluation is stored
    '''    
    
    def __init__(self,multiplicity,time):
        self.multiplicity = multiplicity
        self.time = time

    def spit_out(self):
        result=[]
        result.append(self.multiplicity)
        result.append(self.time)
        
        return result
    '''
    Returns
    -------
    
    result
        A list that contains multiplicity and time [mult,time]
   
    '''



class SliceTimeMult:
    '''
    TimeMultPack unpacks together multiplicity and evaluation time from the PackTimeMult object.
    
    Paramaters
    ---------

    a_list : list
        a_list is the PackTimeMult object

    Attributes
    ----------

    a_list : list
        This stores the list
    '''   


    def __init__(self,a_list):
        self.a_list = a_list


    def get_time(self):
        a_time = self.a_list[1]
        return a_time
    

    def get_multiplicity(self):
        a_multiplicity = self.a_list[0]
        return  a_multiplicity
    '''
    Returns
    -------
    
    a_time
        Retruns just the time

    a_multiplicity
        Returns just the multiplicity
   
    '''


