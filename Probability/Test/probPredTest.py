""" This is the main test for prob.py.  It uses the data generator: 
    Probability/Test/models/probPredTestDat.py.
    In order to run, you must first generate the test data using
    python3 synth/synthDataGen.py Probability/Test/models/probTestDat.py <numRecords>.
    We typically test with 100,000 records, so that is the recommended value
    for numRecords.
"""
import sys
if '.' not in sys.path:
    sys.path.append('.')
from Probability.Prob import ProbSpace
from synth import getData
import sys
import time

def run(filename):
    r = getData.DataReader(filename)
    dat = r.read()
    start = time.time()
    # split data between 'training' and test
    vars = list(dat.keys())
    datLen = len(dat[vars[0]])
    trainLen = datLen - 100
    tr = {}
    te = {}
    for var in dat.keys():
        datL = list(dat[var])
        tr[var] = datL[:trainLen]
        te[var] = datL[trainLen:]
    #print('te = ',te.keys(), te)
    print()
    print ('Testing probability module\'s prediction capabilities.')
    ps = ProbSpace(tr, density=1, power=1)
    print()
    print('Testing non-linear regression with continuous variables.')
    d = ps.distr('Y')
    print('stats(Y) = ', d.mean(), d.stDev(), d.skew(), d.kurtosis())
    # Note: Predict will automatically remove Y from the test data
    Ymean = d.mean()
    expected = te['Y']
    results = ps.Predict('Y', te)
    #print('results = ', results)
    SSE = 0.0 # Sum of squared error
    SST = 0.0 # Sum of squared deviation
    for i in range(len(results)):
        val = results[i]
        exp = expected[i]
        X = []
        for x in ['X1', 'X2', 'X3']:
            X.append(te[x][i])
    
        #print('X = ', X, ', pred = ', val, ', expected = ', exp, ', err = ', val - exp)
        SSE += (val - exp)**2
        SST += (exp - Ymean)**2
    print('R2 = ', 1 - SSE / SST)
    print()
    print('Testing Classification with discontinuous discrete data')
    d = ps.distr('DY')
    print('stats(DY) = ', d.minVal(), d.maxVal(), d.mean(), d.stDev(), d.skew(), d.kurtosis())
    expected = te['DY']
    results = ps.Classify('DY', te)
    #print('results = ', results)
    cumErr = 0
    for i in range(len(results)):
        val = results[i]
        exp = expected[i]
        X = []
        for x in ['DX1', 'DX2', 'DX3', 'DX4']:
            X.append(te[x][i])
    
        #print('X = ', X, ', pred = ', val, ', expected = ', exp, ', err = ', val != exp)
        if val != exp:
            cumErr += 1
    print('Accuracy = ', 1 - (cumErr / len(results)))

    end = time.time()
    duration = end - start
    print('Test Time = ', round(duration))



if __name__ == '__main__':
    filename = 'Probability/Test/models/probPredTestDat.csv'
    run(filename)
