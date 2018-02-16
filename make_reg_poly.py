import numpy as np
from sklearn import neighbors
import os 
import csv
from sklearn.metrics import r2_score, mean_squared_error
X_LIST = [1000,50000,100000, 1000000]
NOISE_LIST = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

np.random.seed(121)
with open('results-polynomial.csv', 'a') as csvfile:
	writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
	for samples in X_LIST:
		directory = "poly/"+str(samples)

		if not os.path.exists(directory):
			os.makedirs(directory)
	    
		for noisy in NOISE_LIST:
			#Sorted list of X-sized list
			X = np.sort(5 * np.random.rand(samples, 1), axis=0)
			#Equally spaced 500 numbers between 0 to 5 that we're going to use for prediction
			T = np.linspace(0, 5, 500)[:, np.newaxis]
			#Take the sine function of values in X as the output
			y = np.sin(X).ravel()
			
			# Add noise to targets
			y[::10] += 1 * (0.5 - np.random.rand(int(samples/10)))

			np.savetxt(directory +"/x-"+str(samples) +"-" + str(noisy) +".txt" , X,  delimiter=',')
			np.savetxt(directory + "/y-"+str(samples) +"-" + str(noisy)+".txt" , y, delimiter=',')
			np.savetxt(directory + "/T-"+str(samples) +"-" + str(noisy)+".txt" , T, delimiter=',')
            

			# #############################################################################
			# Fit regression model
