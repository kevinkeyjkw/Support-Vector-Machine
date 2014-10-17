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

def getSteps(n,kernelMatrix):
	stepsize=[]
	for k in range(n):
		stepsize.append(1/kernelMatrix[k][k])
	return stepsize

def svmDual(n,kernelMatrix,stepsize,y,c):
	alpha0 = np.array([0.0 for i in range(n)])
	alpha1 = np.array([0.0 for i in range(n)])
	kernelMatrix=np.array(kernelMatrix)
	y=np.array(y)
	t = 0
	while True:
		for k in range(n):
			#alpha1[k] = alpha1[k] + stepsize[k]*(1-y[k]*summatio(alpha1,y,kernelMatrix,k,n))
			alpha1[k] = alpha1[k] + stepsize[k]*(1-y[k]*np.dot(alpha1*y,kernelMatrix[:,k]))
			if alpha1[k] < 0: alpha1[k] = 0
			if alpha1[k] > c: alpha1[k] = c
		if np.linalg.norm(alpha1-alpha0) <= epsilon:
			break
		else:
			alpha0 = [x for x in alpha1]
	return alpha1

def getXvect(rows):
	xVect=[]
	for i in range(len(rows)):
		tmp=[a for a in rows[i][:2]]
		tmp.append(1)
		xVect.append(tmp)
	return xVect
# def calculateBias(y,n,weightVec,xVect):
# 	b=[]
# 	#xVect=[a[:2] for a in xVect]
# 	xVect=np.array(xVect)
# 	y = [float(a) for a in y]
# 	for i in range(n):
# 		b.append(y[i]-np.dot(weightVec,xVect[i][:2] ) )
# 	bias = np.mean(b)
# 	return bias

def kernel(rows):
	kernel = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			kernel[i][j] = (np.dot(rows[i],rows[j]))
	return kernel

def calculateWeightVector(alpha,y,xVect):
	weightVect=np.dot(np.array(alpha)*np.array(y),xVect)
	return weightVect

def printSupportVectors(alpha1):
	print('Support vector indicies and alphas:\n')
	count=0
	for a in range(len(alpha1)):
		tmp = alpha1[a]
		if tmp != 0:
			print(a,tmp)
			count = count + 1
	return count
def readData(f):
	rows=[]
	for x in f.readlines():
		temp = x.split(',')
		temp = [float(a) for a in temp]
		temp.append(1)
		rows.append(temp)
	return rows

def splitClasses(rows):
	class1=[]
	class2=[]
	for x in rows:
		if x[2]==1:
			class1.append(x[:2])
		else:
			class2.append(x[:2])
	return np.array(class1),np.array(class2)

f = open(sys.argv[1],'r')
c = int(sys.argv[2])
epsilon = 0.0001
rows=readData(f)
n = len(rows)
y = [x[2] for x in rows]


class1,class2=splitClasses(rows)

plt.plot(class1[:,0],class1[:,1],'^',markersize=7,color='red',alpha=0.5,label='class1')
plt.plot(class2[:,0],class2[:,1],'o',markersize=7,color='blue',alpha=0.5,label='class2')


xVect=getXvect(rows)
#Get kernel
kernelMatrix = kernel(xVect)
#Set step size
stepsize=getSteps(n,kernelMatrix)
#Get alpha
alpha=svmDual(n,kernelMatrix,stepsize,y,c)
#Weight vector
wv = calculateWeightVector(alpha,y,xVect)
#b2=calculateBias([1,1,1,-1,-1],5,np.array([0.833,0.334]), [[3.5,4.25],[4,3],[4.5,1.75],[2,2],[2.5,0.75]])
# x=np.linspace(4,8,150)
# y=(wv[0]*x+wv[2])/-wv[1]
# plt.plot(x,y)
# plt.show()

#Print sv

print('Number of support vectors:',printSupportVectors(alpha))
print('Hyperplane weight vector and bias:',wv)




