import numpy as np
import math as math
import matplotlib.pyplot as plt
import pdb
import sys

def m(D):
    def phi(x):
        triu = np.triu(np.outer(x,x))
        diag = np.diag(np.diag(triu))
        mapped = diag + np.sqrt(2) * (triu - diag)
        return mapped[np.triu_indices_from(mapped)]
    tD = np.asarray([phi(x) for x in D])
    centered = tD - tD.mean(axis=0)
    norms = np.linalg.norm(centered, axis = 1)
    normalized = np.asarray([m/n for (m,n) in zip(centered, norms)])
    nKmap = np.dot(normalized, normalized.T)
    return nKmap

def gradientDir(alpha,rows,kernel,k):
	sum=0
	for i in range(len(rows)):
		sum = sum + alpha[i]*rows[i][2]*kernel[i][k]
	return 1-rows[k][2]*sum	

f = open(sys.argv[1],'r')
c = int(sys.argv[2])
epsilon = 0.0001

rows=[]
for x in f.readlines():
	temp = x.split(',')
	temp = [float(a) for a in temp]
	temp.append(1) #map to one dimension higher?
	rows.append(temp)

#Get kernel
kernelMatrix = m(rows)
#Set step size
n = len(rows)
stepsize=[]
for k in range(n):
	stepsize.append(1/kernelMatrix[k][k])

alpha0 = [0 for i in range(n)]
alpha1 =alpha0
t = 0
while True:
	for k in range(n):
		alpha1[k] = alpha1[k] + stepsize[k]*gradientDir(alpha1,rows,kernelMatrix,k)
		if alpha1[k] < 0: alpha1[k] = 0
		if alpha1[k] > c: alpha1[k] = c

	if np.linalg.norm(np.array(alpha1)-np.array(alpha0)) <= epsilon:
		break
	else:
		alpha0 = alpha1


print('Support vector indicies and alphas:\n')
for a in range(len(alpha1)):
	tmp = alpha1[a]
	if tmp != 0:
		print(a,tmp)

print('Number of support vectors:')
print('Hyperplane weight vector and bias:')




