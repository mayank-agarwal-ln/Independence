from math import e, sqrt, pi, log
import sys
if '.' not in sys.path:
    sys.path.append('.')
import time
from synth import getData
import matplotlib.pyplot as plt
from RKHSmod.rkhs import RKHS

def kExp(x1, x2=0, kparms=[]):
    sigma = kparms[0]
    diff = x1 - x2
    return e**(-((diff)**2) / (2 * sigma**2)) / (sigma * sqrt(2*pi))

def fExp(x, X, kparms):
    sum = 0
    # sigma = kparams[0]
    for xi in X:
        sum += kExp(x, xi,kparms)
    return (sum/len(X))


def exponential(l, x):
    return l*(e**(-1*l*x))

# def exponentialCDF(l,x):
#     return (1-(e**(-1*l*x)))

def testExp(x):
    return exponential(0.12,x)

# def testExpCDF(x):
#     return exponentialCDF(0.4,x)

def logistic(x, mu, s):
    return e**(-(x-mu)/s) / (s * ((1 + e**(-(x-mu)/s)))**2)


def testF(x):
    result = ((exponential(0.5,x)) + logistic(x,5,2))/2 
    return result

if __name__ == '__main__':
    args = sys.argv
    if (len(args) > 1):
        test = args[1]
    else:
        test = 'models/myRKHSTest.csv'
    d = getData.DataReader(test)
    data = d.read()
    X = data['X']
    expmean = sum(X) / len(X)  # to calculate the sigma values
    traces = []  #TODO: What is traces - list of list containing the F(x) values for all testPoints for each sigma
    dataSizes = [50, 100, 1000, 10000, 100000]
    errs = {}
    maxDeviations = {}
    avgDeviations = {}
    testPoints = []
    testMin = -5
    testMax = 5
    tp = testMin
    numTP = 200
    interval = (testMax - testMin) / numTP
    tfs = []
    means = {} 
    for i in range(numTP + 1):
        testPoints.append(tp)
        tfp = testF(tp)
        tfs.append(tfp)
        tp += interval
    delta = 2
    start = time.time()
    for size in dataSizes:
        sigma = 1 / log(size, 4)
        
        r1 = RKHS(X[:size], k=kExp, f=fExp, kparms=[sigma])
        fs = []  # The results using a pdf kernel - RETURNED
        
        totalErr = 0
        deviations = []
        for i in range(len(testPoints)):
            p = testPoints[i]
            fp = r1.F(p)
            fs.append(fp)
            tfp = tfs[i]
            
            err = abs(fp - tfp)
            totalErr += err
        errs[size] = totalErr / numTP
        traces.append(fs) # pdf trace
    print('Average Errors = ', errs)
    end = time.time()
    print('elapsed = ', end - start)
    for t in range(len(traces)):
        fs = traces[t]
        size = dataSizes[int(t)]
        label = 'ExponentialandLogisiticPDF(X)-size=' + str(size)
        linestyle = 'dashed'
        plt.plot(testPoints, fs, label=label, linestyle=linestyle)
    plt.plot(testPoints, tfs, label='testPDF(x)', color='#000000', linewidth=3, linestyle='solid')
    
    plt.legend()
    plt.show()

