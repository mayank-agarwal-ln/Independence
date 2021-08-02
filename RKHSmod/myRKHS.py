from math import e, sqrt, pi, erf, erfc, log, atan
from mpmath import cot
from scipy import signal
# from scipy.special import erfinv
import sys
if '.' not in sys.path:
    sys.path.append('.')
import time
from synth import getData
import matplotlib.pyplot as plt
from RKHSmod.rkhs import RKHS

# def kcdf(x1, x2=0, kparms=[]):
#     # CDF Kernel
#     sigma = kparms[0]
#     return (1 + erf(abs(x2-x1)/sigma*sqrt(2))) / 2

# def Fcdf(x,X, kparms):
#     sigma = kparms[0]
#     delta = kparms[1]
#     cum = 0.0
#     for p in X:
#         if p < x:
#             if delta is not None and (x-p) > delta * sigma:
#                 cum += 1
#                 continue
#             cum += kcdf(p, x, kparms)
#         else:
#             if delta is not None and (p-x) > delta * sigma:
#                 continue
#             cum += 1 - kcdf(p, x, kparms)
#     return cum / len(X)

def kExp(x1, x2=0, kparms=[]):
    sigma = kparms[0]
    diff = x1 - x2
    return e**(-((diff)**2) / (2 * sigma**2)) / (sigma * sqrt(2*pi))

def kSaw(x1, x2=0, kparms=[]):
    diff = x1-x2
    return signal.sawtooth(diff)

def kSaw2(x1,x2,kparms=[]):
    # a = kparms[0]   <- amplitude 
    # p = kparms[1]   <- period
    a = 1
    p = 1
    diff = x1-x2
    res = -1*(2/pi)*(atan(cot((diff*pi)/p)))
    return res

def fExp(x, X, kparms):
    sum = 0
    # sigma = kparams[0]
    for xi in X:
        sum += kSaw2(x,xi,kparms)
        # sum += kExp(x, xi,kparms)
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

# def logisticCDF(x, mu, s):
#     return 1 / (1 + e**(-(x - mu)/s))

def testF(x):
    # result = (logistic(x, -2, .5) + logistic(x, 2, .5)  + logistic(x, -.5, .3)) / 3
    # result = ((exponential(1,x)) + logistic(x,-2,1))/2 
    result = logistic(x,0,1)
    return result

# def testFCDF(x):
#     result = (logisticCDF(x, -2, .5) + logisticCDF(x, 2, .5) + logisticCDF(x, -.5, .3)) / 3
#     return result

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
    # dataSizes = [10000]
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
    # ctfs = []
    means = {} 
    # Generate a uniform range of test points.
    # While at it, generate our expected pdf and cdf
    for i in range(numTP + 1):
        testPoints.append(tp)
        # tfp = testExp(tp)  # EXPECTED Exponential
        tfp = testF(tp)
        tfs.append(tfp)
        # ctfp = testExpCDF(tp) # EXPECTED Exponential CDF
        # ctfs.append(ctfp)
        tp += interval
    delta = 2
    start = time.time()
    for size in dataSizes:
        # Choose a reasonable sigma based on data size.
        sigma = 1 / log(size, 4)
        # r1 = RKHS(X[:size], k=kExp, f=fExp, kparms=[sigma])
        r1 = RKHS(X[:size], k=kExp, f=fExp, kparms=[sigma])
        fs = []  # The results using a pdf kernel - RETURNED
        # fsc = [] # Results using a cdf kernel - RETURNED
        totalErr = 0
        deviations = []
        for i in range(len(testPoints)):
            p = testPoints[i]
            fp = r1.F(p)
            fs.append(fp)
            # fpc = r2.F(p)
            # fsc.append(fpc)
            tfp = tfs[i]
            # ctfp = ctfs[i]
            err = abs(fp - tfp)
            totalErr += err
            #print('fpc, ctfp = ', fpc, ctfp)
            # deviation = abs(fpc - ctfp)
            # deviations.append(deviation)
        # maxDeviations[size] = max(deviations)
        # avgDeviations[size] = sum(deviations) / numTP
        errs[size] = totalErr / numTP
        traces.append(fs) # pdf trace
        # traces.append(fsc) # cdf trace
        # r3 = RKHS(X[:size], f=Fquant, k=kQuant, kparms=[sigma, None]) #TODO: What is Fquant and Kquant
        # mean = r3.F(.5)
        # means[size] = mean
    print('Average Errors = ', errs)
    # print('Average Deviation = ', avgDeviations)
    # print('Maximum Deviation = ', maxDeviations)
    # print('Means = ', means, expmean)
    end = time.time()
    print('elapsed = ', end - start)
    for t in range(len(traces)):
        fs = traces[t]
        # size = dataSizes[int(t/2)] # traces are alternately pdf and cdf
        # if t%2 == 0:
        #     label = 'pdf(X)-size=' + str(size)
        #     linestyle = 'solid'
        # else:
        #     label = 'cdf(X)-size=' + str(size)
        #     linestyle = 'dashed'
        size = dataSizes[int(t)]
        label = 'ExponentialandLogisiticPDF(X)-size=' + str(size)
        linestyle = 'dashed'
        plt.plot(testPoints, fs, label=label, linestyle=linestyle)
    plt.plot(testPoints, tfs, label='testPDF(x)', color='#000000', linewidth=3, linestyle='solid')
    # plt.plot(testPoints, ctfs, label='testCDF(x)', color='#000000', linewidth=3, linestyle='dotted')
    plt.legend()
    plt.show()

