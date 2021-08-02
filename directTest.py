import sys
import rv
import cGraph
from synth import getData
import independence
import time

args = sys.argv
if (len(args) > 1):
    test = args[1]

f = open(test, 'r')
exec(f.read(), globals())

print('Testing: ', test, '--', testDescript)
start = time.time()
gnodes = []

# 'model' is set when the text file is exec'ed
for var in model:
    observed = True
    dType = 'Numeric'
    name, parents = var[:2]
    if len(var) >= 3:
        observed = var[2]
    if len(var) >= 4:
        dType = var[3]
    gnode = rv.rv(name, parents, observed, dType, None, None)
    gnodes.append(gnode)

# For dat file, use the input file name with the .csv extension
tokens = test.split('.')
testFileRoot = str.join('.',tokens[:-1])
datFileName = testFileRoot + '.csv'

d = getData.DataReader(datFileName)
data = d.read()

g = cGraph.cGraph(gnodes, data)

g.printGraph()

g.testDirection()

g.findExogenous()


end = time.time()
duration = end - start
print('Test Time = ', round(duration))