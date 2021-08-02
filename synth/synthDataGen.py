import numpy as np
import math
from numpy.random import *
import sys
import random

# Tuning Parameters
MIN_COEF = .1  # Minimum absolute value for coefficients
COEF_SCALE = 5 # Scale of exponential distribution to be added to the MIN_COEF
MEAN_MEAN = 0.0  # Average value of generated noise means
MEAN_SCALE = 30 # Gaussian Standard Deviation to be added to MEAN_MEAN to choose the Mean of the noise
MIN_STD = .1 # Minimum Scale parameter to be applied when generating noise
STD_SCALE = 5 # Scale of exponential distribution to be added to MIN_STD when chosing the scale of the noise.

#DISTRS = ['normal'] # Set of distributions to be chosen from for generating noise.  
						# Note distributions must support location and scale parameters, and must allow negative locations

						
DISTRS = ['lognormal', 'laplace', 'logistic', 'gumbel']
VALIDATION = None
CURR_EQUATION = 0
NOISE_COUNT = 0
NOISES = {}
COEF_COUNT = 0
COEFS = {}
RAW_EQUATIONS = []
MAX_DIFFICULTY = 0

# Input file name should typically have a .py extension
def run(semFileName, samples=1000, maxDifficulty=0, reset=False):
    global VALIDATION, NOISES, NOISE_COUNT, COEFS, COEF_COUNT, CURR_EQUATION, RAW_EQUATIONS
    global smallestStd, largestStd, smallestCoef, largestCoef, MAX_DIFFICULTY
    if not RAW_EQUATIONS:
        # First time.  Treat as if reset is True
        reset = True
    f = open(semFileName, 'r')
    contents = f.read()
    exec(contents, globals())
    # For out file, use the input file name with the .csv extension
    if maxDifficulty:
        MAX_DIFFICULTY = maxDifficulty
    tokens = semFileName.split('.')
    outFileRoot = str.join('.',tokens[:-1])
    outFileName = outFileRoot + '.csv'
    RAW_EQUATIONS = varEquations # From tag in the SEM file 'varEquations', a list of equations
    varNames = []
    for rv in model:
        observed = True
        if type(rv) == type((1,)):
            if len(rv) >= 2:
                name, parents = rv[:2]
            if len(rv) >= 3:
                observed = rv[2]
            if len(rv) >= 4:
                datType = rv[3]
        else:
            # Just a list of var names.  No model.
            name = rv
            parents = []
        if observed:
            varNames.append(name)
    success = False
    while not success:
        outLines = []
        cEquations = []
        cVarNames = []
        #print('reset = ', reset)
        if reset:
            NOISES = {}
            COEFS = {}
            smallestStd = 10**100
            largestStd = 0
            smallestCoef = 10**100
            largestCoef = 0
        for eq in varEquations:
            cEquations.append(compile(eq,'err', 'single'))
        for varName in varNames:
            cVarNames.append(compile(varName, 'err', 'eval'))

        outLine = str.join(',', varNames) + '\n'
        outLines.append(outLine)


        for sample in range(samples):
            outTokens = []
            NOISE_COUNT = 0
            COEF_COUNT = 0
            for i in range(len(cEquations)):
                CURR_EQUATION = i
                varEquation = cEquations[i]
                try:
                    exec(varEquation)
                except:
                    print('*** Invalid Equation = ', RAW_EQUATIONS[i])
                    print(getSEM())
            for i in range(len(cVarNames)):
                outTokens.append(str(eval(cVarNames[i])))
                #print (varNames[i], '=', eval(cVarNames[i]))
            endline = '\n'	
            if sample == samples-1:
                endline = ''
            outLine = str.join(',', outTokens) + endline
            outLines.append(outLine)
        f = open(outFileName, 'w')
        f.writelines(outLines)
        f.close()
        success = True
    return outFileName

def determineDifficulty(fileName, samples):
	import analyzeSEM
	sa = analyzeSEM.SemAnalyzer(fileName)
	stats = sa.analyzeOne(samples, gen=False, valData=getValidation())
	diff = stats['normDifficulty']
	#print('sem = ', getSEM())
	#print('difficulty = ', diff)
	return diff

def increaseRange():
	global COEF_SCALE, STD_SCALE, MEAN_SCALE
	COEF_SCALE *= 1.1
	STD_SCALE *= 1.1
	MEAN_SCALE *= 1.1
	return

def decreaseRange():
	global COEF_SCALE, STD_SCALE, MEAN_SCALE
	COEF_SCALE *= .9
	STD_SCALE *= .9
	MEAN_SCALE *= .9
	return
	
def getSEM():
    global NOISE_COUNT, COEF_COUNT
    NOISE_COUNT = 0
    COEF_COUNT = 0
    outEquations = []
    for equation in RAW_EQUATIONS:
        equation = "'" + equation + "',"
        while equation.find('data()') >= 0:
            equation = equation.replace('data()', str(DATA_OFFSET) + ' + ' + NOISES[NOISE_COUNT],1)
            temp = noise() # to increment the NOISE_COUNT
        while equation.find('coef()') >= 0:
            coefVal = coef()
            equation = equation.replace('coef()',str(coefVal),1)
        while equation.find('noise()') >= 0:
            equation = equation.replace('noise()', NOISES[NOISE_COUNT],1)
            temp = noise() # to increment the NOISE_COUNT
        outEquations.append(equation)
    if largestCoef > 0 or largestStd > 0:
        outEquations.append('Stats:')
        coefRange = largestCoef / float(smallestCoef)
        stdRange = largestStd / float(smallestStd)
        totalRange = coefRange * stdRange
        outEquations.append('  Coef Range = ' + str(coefRange))
        outEquations.append('  Std Range = ' +str(stdRange))
        outEquations.append('  Total Scale Range = ' + str(totalRange))
    outStr = '\n' + str.join('\n', outEquations) + '\n'
    #print(outEquations, outStr)
    return outStr
	
#DISTRS = ['logistic']
#DISTRS = ['logistic', 'lognormal', 'laplace','gumbel']

def noise():
	# Generate a random noise distribution and return a sample from it
	global NOISE_COUNT
	currNoise = NOISES.get(NOISE_COUNT, None)
	if currNoise is None:
		distType = random.choice(DISTRS)
		mean = chooseMean()
		std = chooseStd()
		currNoise = distType + '(' + str(mean) + ',' + str(std) + ')'
		NOISES[NOISE_COUNT] = currNoise
		print('noise[', NOISE_COUNT, '] = ', currNoise)
	NOISE_COUNT += 1
	return eval(currNoise)

DATA_OFFSET = 1.000

def data():
	data = noise() + DATA_OFFSET
	return data
	
# Stats:
smallestStd = 10**100
largestStd = 0
smallestCoef = 10**100
largestCoef = 0


def coef():
	global COEF_COUNT
	global smallestCoef, largestCoef
	coef = COEFS.get(COEF_COUNT, None)
	if coef is None:
		coef = exponential(COEF_SCALE*MIN_COEF) + MIN_COEF
		if coef < smallestCoef:
			smallestCoef = coef
		if coef > largestCoef:
			largestCoef = coef
		sign = random.choice([-1,1,1,1,1])
		coef *= sign
		coef = round(coef, 3)
		COEFS[COEF_COUNT] = coef
		print('coef[', COEF_COUNT, '] = ', coef)
	COEF_COUNT += 1
	return coef



def chooseMean():
	return round(normal(MEAN_MEAN, MEAN_SCALE),3)


def chooseStd():
	global smallestStd, largestStd
	val = round(exponential(STD_SCALE * MIN_STD) + MIN_STD,3)
	if val < smallestStd:
		smallestStd = val
	if val > largestStd:
		largestStd = val
	return val
	

def getValidation():
	return VALIDATION



if __name__ == '__main__':
	filename = None
	if len(sys.argv) > 1:
		filename = sys.argv[1]
	else:
		print ('first parameter is the SEM filename -- Required')
	if len(sys.argv) > 2:
		datacount = eval(sys.argv[2])
	else:
		datacount = 1000
	print('Generating data for model = ', filename)
	#print('filename, datacount = ', filename, datacount)
	if filename is not None:
		run(filename, samples=datacount, maxDifficulty=0)
	print ('SEM = ', getSEM())
	print('Generated ', datacount, 'records.')

	