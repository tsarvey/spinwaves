class ParamManager():
    def __init__(self):
        self.parameters = []
    
    def addParam(self, param):
        """Adds a new parameter to this list."""
        self.parameters.append(param)
        
    def validIndex(self, index):
        """Checks if the integer index is a valid index(identifier) of a parameter."""
        return (index < len(self.parameters))
    
    def tie(self, paramObj, param2_index):
        index1 = self.parameters.index(paramObj)
        #Add second index to first param
        paramObj.tied.append(param2_index)
        #Add first index to second param
        self.parameters[param2_index].tied.append(index1)
        
    def removeParam(self, param):
        """removes the given JParam object from the list of parameters and corrects all
        indices and tied lists."""
        
        index = self.parameters.index(param)
        self.parameters.pop(index)
        for parameter in self.parameters:
            for tiedIndex in parameter.tied:
                if tiedIndex>=index:
                    tiedIndex-=1
            