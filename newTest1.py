import sys
import rv
import cGraph
from synth import getData
import independence
from Probability import Prob
from synth import synthDataGen
from standardize import standardize
import time


# METHOD = 'prob'
METHOD = 'rcot'
POWER = 1
print('power = ', POWER)
args = sys.argv
test = 'models/nCondition.csv'

r = getData.DataReader(test)
dat = r.read()
vars = dat.keys()
for var in vars:
    dat[var] = standardize(dat[var])
ps = Prob.ProbSpace(dat, power=POWER)

# List a variety of independent relationships
indeps = [
        ('B', 'C'),
        ('D', 'C'),
        ('F', 'E'),
        ('B', 'F'),
        ('E', 'C'),
        ('A2','A4',['B','C','D']),
          ]

# List a variety of dependent relationships
deps = [
        ('A2', 'B'),
        ('A2','A3',['B']),
        ('A2','A3',['B','C']),
        ('A2','A3',['B']),
        ('A2','A3',['D']),
        ('A3','A4',['B','C','D']),
        ('A3','A4',['B','C']),
        ('A3','A4',['B']),
        ('A3','A4',['E','F']),
        ('A4','A5',['B','C','D','E']),
        ('A5','A6',['B','C','D','E']),
        ('C', 'B',['A3']),
        ('F', 'B',['A3','A4','A5','A6']),
        ]
print('Testing: ', test)
start = time.time()

testVal = 0
condTestVal = 0
delta = .1

num_f = 25 
num_f2 = 5
r = 500

minIndep = 999999.0
maxDep = 0.0
minDep = 9999999.0
cumIndep = 0.0
cumDep = 0.0
print()
# print('Testing expected independents:')
for ind in indeps:
    if len(ind) == 2:
        x, y = ind
        z = []
    elif len(ind) == 3:
        x, y, z = ind
    else:
        print('* Error, improperly specified independence =', ind)
    pval = independence.test(ps, [x], [y], z, METHOD, POWER,num_f,num_f2,r)
    if pval < minIndep:
        minIndep = pval
    print('dependence', ind, '= ', 1-pval)

print()
# print('Testing expected dependents:')
# print()
for dep in deps:
    if len(dep) == 2:
        x, y = dep
        z = []
    elif len(dep) == 3:
        x, y, z = dep
    else:
        print('* Error, improperly specified independence =', dep)
    pval = independence.test(ps, [x], [y], z, METHOD, POWER,num_f,num_f2,r)
    if pval > maxDep:
        maxDep = pval
    if pval < minDep:
        minDep = pval
    print('dependence', dep, ' = ', 1-pval)
print()

print('Maximum dependence for expected independents = ', 1-minIndep)
print('Minimum dependence for expected dependents =', 1 - maxDep)
print('Margin = ', minIndep - maxDep, '.  Positive margin is good.')
print('Maximum dependence = ', 1-minDep)
print('best Low threshold is: ', max(
    [((1-minIndep) + (1 - maxDep)) / 2.0, 1-minIndep + .001]))
print('best High threshold is: ', 1 - minDep + .1)
print()
end = time.time()
duration = end - start
print('Test Time = ', round(duration))