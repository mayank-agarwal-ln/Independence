import numpy as np
import random


class DataReader():
    def __init__(self, input, limit=0):
        if limit == 0:
            self.limit = 100000000
        else:
            self.limit = limit
        self.varData = {}
        self.vars = []
        self.varIndex = {}
        self.sampleCount = 0
        f = open(input, 'r')
        lines = f.readlines()
        varNames = lines[0]
        data = lines[1:]
        tokens = varNames[:-1].split(',')
        for varName in tokens:
            self.vars.append(varName)
            self.varData[varName] = []
            self.varIndex[varName] = len(self.vars) - 1
        if len(data) < limit:
            print('*** Number of datapoints is less than requested limit (', len(data), ' vs ', self.limit, ') -- Using data length')
        # If limit is less than length of data, select a random starting point that will produce enough (i.e. limit) datapoints
        if len(data) > self.limit:
            datalen = len(lines)
            dataslack = datalen - self.limit
            datastart = random.choice(range(dataslack))
            data = data[datastart:datastart + self.limit]
        for line in data:
            if line[-1] == '\n':
                line = line[:-1]
            tokens = line.split(',')
            for i in range(len(self.vars)):
                val = float(tokens[i])
                self.varData[self.vars[i]].append(val)
        self.sampleCount = len(self.varData[self.vars[0]])
        print('getData: ', len(data), 'records read.')
        #np.random.shuffle(self.vars)

    def read(self):
        return self.varData
		
		
    def getSeriesNames(self):
        return self.vars[:]
		
    def getSeries(self, varName):
        return self.varData[varName][:self.limit]
		
    def getIndexForSeries(self, varName):
        return self.varIndex[varName]
			
	

				
		