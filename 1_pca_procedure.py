import numpy as np
import scipy as sc
from numpy.linalg import inv
from numpy import linalg as LA
import corner
from matplotlib import pyplot as pl

#Data
chain2 = sc.genfromtxt('/home/williams/Downloads/Dissertation/Final_Results/all_data/outputs2/chain.txt', usecols = (0,1,2,3,4,5,25,26,27,28,29,30))
chain3 = sc.genfromtxt('/home/williams/Downloads/Dissertation/Final_Results/all_data/outputs3/chain.txt', usecols = (0,1,2,3,4,5,25,26,27,28,29,30))
chain4 = sc.genfromtxt('/home/williams/Downloads/Dissertation/Final_Results/all_data/outputs4/chain.txt', usecols = (0,1,2,3,4,5,25,26,27,28,29,30))
chain5 = sc.genfromtxt('/home/williams/Downloads/Dissertation/Final_Results/all_data/outputs5/chain.txt', usecols = (0,1,2,3,4,5,25,26,27,28,29,30))
chain = np.vstack((chain3,chain4))
chain = np.vstack((chain2,chain))
chain = np.vstack((chain,chain5))
deriv_param = sc.genfromtxt('/home/williams/Downloads/Dissertation/Final_Results/all_data/outputs2/derived_params.txt', usecols = (0,1,3))
deriv_param = np.vstack((deriv_param,sc.genfromtxt('/home/williams/Downloads/Dissertation/Final_Results/all_data/outputs3/derived_params.txt', usecols = (0,1,3))))
deriv_param = np.vstack((deriv_param,sc.genfromtxt('/home/williams/Downloads/Dissertation/Final_Results/all_data/outputs4/derived_params.txt', usecols = (0,1,3))))
deriv_param = np.vstack((deriv_param,sc.genfromtxt('/home/williams/Downloads/Dissertation/Final_Results/all_data/outputs5/derived_params.txt', usecols = (0,1,3))))


#Column positions in chain for the wi's
col_min = 6
col_max = 12 #this number is actually col_max+1
pos_wi = np.arange(col_min, col_max)

#Averages of eq. of state parameters
w1 = np.average(chain[:, pos_wi[0]])
w2 = np.average(chain[:, pos_wi[1]])
w3 = np.average(chain[:, pos_wi[2]])
w4 = np.average(chain[:, pos_wi[3]])
w5 = np.average(chain[:, pos_wi[4]])
w6 = np.average(chain[:, pos_wi[5]])

####################################################################################################################################
#PRINCIPAL COMPONET ANALYSIS

#Evaluating the covariance matrix
first_term = sc.array(sc.zeros(6*6).reshape(6,6))
for i in range(chain.shape[0]):
    first_term += np.outer(chain[i,col_min:col_max],chain[i,col_min:col_max])/chain.shape[0]

covmat = first_term - np.outer(np.array([w1,w2,w3,w4,w5,w6]),np.array([w1,w2,w3,w4,w5,w6]))
covmatnumpy = np.cov(chain[:,col_min:col_max].T)

#Check covariance matrices evaluated using different methods
if (np.max(np.abs(covmat-covmatnumpy))<1e-03):
    print('Covariance matrix OK')
else:
    print('Something is wrong with covariance matrix!')

#Eigenvalues and eigenvectors
invcovmat = inv(covmat)
eigvalues, eigvectors = LA.eig(invcovmat)
eigvalues = np.diag(eigvalues)
inveigvalues = inv(eigvalues)
inveigvectors = inv(eigvectors)

#Transformation matrix (weights)
transmatrix = np.dot(eigvectors, np.dot(np.sqrt(eigvalues), eigvectors.T))

#Normalization so that the rows of transmatrix sum to one
for i in range(transmatrix.shape[0]):
    transmatrix[i,:] = transmatrix[i,:]/np.sum(transmatrix[i,:])

testeinvcovmat = np.dot(eigvectors, np.dot((eigvalues), eigvectors.T))

#Check inverse of covariance matrices
if (np.max(np.abs(invcovmat-testeinvcovmat))<1e-03):
    print('Inverse covariance matrix OK')
else:
    print('Something is wrong with inverse covariance matrix!')


chain_new = np.copy(chain) #np.copy because otherwise chain_new becomes a pointer to chain

#Evaluating the new table of w_i parameters using the transformation matrix
for i in range(chain.shape[0]):
    chain_new[i,col_min:col_max] = np.dot(transmatrix, chain[i,col_min:col_max])


#Testing if the new covariance matrix is diagonal
w1 = np.average(chain_new[:, pos_wi[0]])
w2 = np.average(chain_new[:, pos_wi[1]])
w3 = np.average(chain_new[:, pos_wi[2]])
w4 = np.average(chain_new[:, pos_wi[3]])
w5 = np.average(chain_new[:, pos_wi[4]])
w6 = np.average(chain_new[:, pos_wi[5]])

first_term = sc.array(sc.zeros(6*6).reshape(6,6))
for i in range(chain_new.shape[0]):
    first_term += np.outer(chain_new[i,col_min:col_max],chain_new[i,col_min:col_max])/chain_new.shape[0]


covmatnew = first_term - np.outer(np.array([w1,w2,w3,w4,w5,w6]),np.array([w1,w2,w3,w4,w5,w6])) #chech if this is diagonal and matches covmatnumpynew
covmatnumpynew = np.cov(chain_new[:,col_min:col_max].T) #chech if this is diagonal and matches covmatnew

#Check if new covariance matrices agree
if (np.max(np.abs(covmatnew-covmatnumpynew))<1e-03):
    print('New covariance matrix OK')
else:
    print('Something is wrong with new covariance matrix!')

#Check if new covariance matrices are diagonal
for i in range(len(covmatnew[:,0])):
    for j in range(len(covmatnew[0,:])):
        if ((i!=j) and (np.abs(covmatnew[i,j]>1e-07))) or ((i!=j) and (np.abs(covmatnumpynew[i,j]>1e-07))):
            print('New covariance matrices are not diagonal!')
            break
print('New covariance matrix is diagonal')
            

np.savetxt('chain_uncorrelated.txt', chain_new)
#print(len(chain[:,0]))
#print(len(chain_new[:,0]))


