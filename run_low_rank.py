import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from beautifultable import BeautifulTable
import csv

table = BeautifulTable()
table.column_headers = ["X", "Y", "NOISE", "K", "Data Load Time", "Train Time", "Test Time", "R2-Train", "R2-Test", "MSE-Train", "MSE_Test"]
X_LIST = [1000,50000,100000]
Y_LIST = [2,15,20,30]
NOISE_LIST = [0,10,50,100]
N_NEIGH = [ 1, 3, 5, 7, 9 ,10 ]
total_time_start = time.time()
with open('low_rank_results.csv', 'a') as csvfile:
	writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
	for samples in X_LIST:
		for features in Y_LIST:
			for noisy in NOISE_LIST:
				for k in N_NEIGH:
					x_path = "lowrank/"+str(samples) + "/x-" + str(samples) + "-" + str(features) + "-" + str(noisy) +".txt"
					y_path = "lowrank/"+ str(samples) + "/y-" + str(samples) + "-" + str(features) + "-" + str(noisy) +".txt"
					start_load_time = time.time()
					x = np.loadtxt(x_path, delimiter=",")
					y = np.loadtxt(y_path , delimiter=",")
					data_load_time = time.time() - start_load_time
					X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

					neigh = KNeighborsRegressor(n_neighbors=k)
					start_train_time = time.time()
					model = neigh.fit(X_train, y_train) 
					train_time = time.time() - start_train_time
					#x1 , y1, coeff = make_regression(n_samples=100, n_features=4, coef=True)
					start_test_time = time.time()
					train_pred = neigh.predict(X_train)
					pred = neigh.predict(X_test)
					test_time = time.time() - start_test_time

					train_r2_score_result = r2_score(y_train, train_pred)
					train_mse_result = mean_squared_error(y_train,train_pred)

					test_r2_score_result = r2_score(y_test, pred)
					test_mse_result = mean_squared_error(y_test,pred)

					table.append_row([samples, features, noisy, k, data_load_time, train_time, test_time, train_r2_score_result, test_r2_score_result, 
train_mse_result, test_mse_result])
					writer.writerow([samples, features, noisy, k, data_load_time, train_time, test_time, train_r2_score_result, test_r2_score_result, 
train_mse_result, test_mse_result])

total_running_time = time.time() - total_time_start
print(total_running_time)					
print(table)
