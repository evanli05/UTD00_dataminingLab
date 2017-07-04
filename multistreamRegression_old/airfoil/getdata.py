import random
import math, numpy as np
from scipy.stats import multivariate_normal
import csv
import pandas
from copy import deepcopy
records = []
with open('target.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # print row
        records.append(row)

# print records
random.shuffle(records)
lenth = len(records)
D1 = records[:lenth/2]
trg = records[lenth/2:]
target = deepcopy(trg)
# print len(target)
##compute covariance
alpha = 0.03
a1 = []
for i in range(0,len(D1)):
   a1.append(float(D1[i][0]))
a2 = []
for i in range(0,len(D1)):
   a2.append(float(D1[i][1]))
a3 = []
for i in range(0,len(D1)):
   a3.append(float(D1[i][2]))
a4 = []
for i in range(0,len(D1)):
   a4.append(float(D1[i][3]))
a5 = []
for i in range(0,len(D1)):
   a5.append(float(D1[i][4]))

x1 = np.array(a1)
x2 = np.array(a2)
x3 = np.array(a3)
x4 = np.array(a4)
x5 = np.array(a5)
Q = np.cov([x1,x2,x3,x4,x5])
Q = alpha*Q
#print Q

xbar = []
xbar.append(np.mean(x1))
xbar.append(np.mean(x2))
xbar.append(np.mean(x3))
xbar.append(np.mean(x4))
xbar.append(np.mean(x5))
#print xbar

seed = random.sample(D1,1)
#print seed[0][0]
xseed = []
for i in range(0, 5):
    xseed.append(float(seed[0][i]))

print xseed

## get weight for data by given distribution

weight = []
for j in range(0,len(D1)):
    instance = []
    for i in range(0,5):
        instance.append(float(D1[j][i]))
    p = multivariate_normal.pdf(instance, xseed, Q,allow_singular=True)
        #print p,instance
    weight.append(float(p))
#weight.sort()
print weight

src = []
#for i in range(0,len(D1)):
#    if weight[i]>6e-10:
#       src.append(D1[i])
#print src
D1len = len(D1)
index = []
for i in range(0,D1len):
    index.append(i)
# print 1
# print index
srcindex = []
# print len(trg)
srcindex = np.random.choice(index,len(target)/9, 'False', weight[:752])
# print weight[:752]
# print srcindex
src = []
for i in range(0,len(srcindex)):
    src.append(D1[srcindex[i]])
# print src
# print trg

my_df = pandas.DataFrame(src)
my_df.to_csv('datasrc.csv', index=False, header=False)

my_df = pandas.DataFrame(trg)
my_df.to_csv('datatrg.csv', index=False, header=False)


