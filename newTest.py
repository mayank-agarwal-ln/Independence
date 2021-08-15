import sys
import rv
import cGraph
from synth import getData
import independence
from Probability import Prob
from synth import synthDataGen
from standardize import standardize
import time

#METHOD = 'prob'
METHOD = 'rcot'
# METHOD = 'fcit'
POWER = 1
#print('power = ', POWER)
args = sys.argv
test = 'Probability/Test/models/indCalibrationDat.csv'

r = getData.DataReader(test)
dat = r.read()
vars = dat.keys()
for var in vars:
    dat[var] = standardize(dat[var])
#print('dat = ', dat)
ps = Prob.ProbSpace(dat, power=POWER)

# List a variety of independent relationships
indeps = [('L1', 'L2'),
          ('L2', 'L3'),
          ('L1', 'L3'),
          ('E1', 'E2'),
          ('N1', 'N2'),
          ('L4', 'L5'),
          ('L5', 'L6'),
          ('L4', 'N3'),
          ('B', 'D', ['A']),
          ('A', 'C', ['B', 'D']),
          ('C', 'E2'),
          ('L6', 'L7', ['L3']),
          ('L4', 'L6', ['L3']),
          ('L8', 'L9', ['L1']),
          ('M1', 'E2'),
          ('M1', 'E2'),
          ]

# List a varieety of dependent relationships
deps = [('L3', 'L4'),
        ('L5', 'L2'),
        ('L6', 'L3'),
        ('L6', 'L7'),
        ('L7', 'L4'),
        ('E3', 'E1'),
        ('E3', 'E2'),
        ('M1', 'N2'),
        ('B', 'D'),
        ('B', 'D', 'C'),
        ('B', 'D', ['A', 'C']),
        ('B', 'A', 'C'),
        ('B', 'A', ['C', 'D']),
        ('B', 'C', 'A'),
        ('A', 'C', 'B'),
        ('L8', 'L9'),
        ('N1', 'N2', ['N3']),
        ('N3', 'E1', ['M1']),
        ]
#print('Testing: ', test)
start = time.time()

testVal = 0
condTestVal = 0
delta = .1

minIndep = 999999.0
maxDep = 0.0
minDep = 9999999.0
cumIndep = 0.0
cumDep = 0.0

numfs = [22,25,28,30,32,35]
numf2s = [2,3,5,6]
rs = [500,600,750,800]


for num_f in numfs:
    for num_f2 in numf2s:
        for r in rs:
            ind_d = {}
            dep_d = {}
            for i in range(25):
                for ind in indeps:
                    if len(ind) == 2:
                        x, y = ind
                        z = []
                    elif len(ind) == 3:
                        x, y, z = ind
                    else:
                        print('*** Error, improperly specified independence =', ind)
                    pval = independence.test(ps, [x], [y], z, METHOD, POWER,num_f,num_f2,r)
                    if pval < minIndep:
                        minIndep = pval
                    key = str(ind)
                    if(key in ind_d):
                        ind_d[key].append(1-pval)
                    else:
                        ind_d[key] = [1-pval]
                for dep in deps:
                    if len(dep) == 2:
                        x, y = dep
                        z = []
                    elif len(dep) == 3:
                        x, y, z = dep
                    else:
                        print('*** Error, improperly specified independence =', dep)
                    pval = independence.test(ps, [x], [y], z, METHOD, POWER,num_f,num_f2,r)
                    if pval > maxDep:
                        maxDep = pval
                    if pval < minDep:
                        minDep = pval
                    key = str(dep)
                    if(key in dep_d):
                        dep_d[key].append(1-pval)
                    else:
                        dep_d[key] = [1-pval]
            print("num_f: ",num_f," num_f2: ",num_f2," r: ",r)
            print()
            print("Independents")
            for i in ind_d.keys():
                print(i, str(sum(ind_d[i])/(len(ind_d[i]))))
            print()
            print("Dependents")
            for i in dep_d.keys():
                print(i, str(sum(dep_d[i])/(len(dep_d[i]))))
            print("--------------------------------------------------------------------------")

# print('Maximum dependence for expected independents = ', 1-minIndep)
# print('Minimum dependence for expected dependents =', 1 - maxDep)
# print('Margin = ', minIndep - maxDep, '.  Positive margin is good.')
# print('Maximum dependence = ', 1-minDep)
# print('best Low threshold is: ', max(
#     [((1-minIndep) + (1 - maxDep)) / 2.0, 1-minIndep + .001]))
# print('best High threshold is: ', 1 - minDep + .1)
# print()
# end = time.time()
# duration = end - start
# print('Test Time = ', round(duration))
