import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from beautifultable import BeautifulTable
import csv

table = BeautifulTable()
table.column_headers = ["X", "NOISE", "K", "Data Load Time", "Train Time", "Test Time", "R2-Train", "R2-Test", "MSE-Train", "MSE_Test"]
X_LIST = [1000,50000,100000]
Y_LIST = [2,15,20,30]
NOISE_LIST = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
N_NEIGH = [ 1, 3, 5, 7, 9 ,10 ]

# X_LIST = [1000]
# Y_LIST = [2]
# NOISE_LIST = [0.1]
# N_NEIGH = [ 1 ]

total_time_start = time.time()
with open('results-polynomial.csv', 'a') as csvfile:
	writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
	for samples in X_LIST:
		for noisy in NOISE_LIST:
			for k in N_NEIGH:
				for i, weights in enumerate(['uniform', 'distance']):
					x_path = "poly/"+str(samples) + "/x-" + str(samples) + "-" + str(noisy) +".txt"
					y_path = "poly/"+str(samples) + "/y-" + str(samples) +  "-" + str(noisy) +".txt"
					t_path = "poly/"+str(samples) + "/T-" + str(samples) +  "-" + str(noisy) +".txt"
					start_load_time = time.time()
					x = np.loadtxt(x_path, delimiter=",", ndmin=2)

					y = np.loadtxt(y_path , delimiter=",", ndmin=2)
					T = np.loadtxt(t_path , delimiter="," ,ndmin=2)
					data_load_time = time.time() - start_load_time
					knn = KNeighborsRegressor(k, weights=weights)

					start_train_time = time.time()
					model = knn.fit(x, y)
					train_time = time.time() - start_train_time
					start_test_time = time.time()
					y_ = model.predict(T)
					test_time = time.time() - start_test_time
					train_pred = knn.predict(x)

					train_r2_score_result = r2_score(y, train_pred)
					train_mse_result = mean_squared_error(y,train_pred)

					test_r2_score_result = r2_score(y_, np.sin(T).ravel())
					test_mse_result = mean_squared_error(y_,np.sin(T).ravel())

					table.append_row([samples, noisy, k, data_load_time, train_time, test_time, train_r2_score_result, test_r2_score_result, 
train_mse_result, test_mse_result])
					writer.writerow([samples, noisy, k, data_load_time, train_time, test_time, train_r2_score_result, test_r2_score_result, 
train_mse_result, test_mse_result])

total_running_time = time.time() - total_time_start
print(total_running_time)					
print(table)



