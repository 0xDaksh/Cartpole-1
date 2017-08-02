import numpy as np
def process_data(training_data):
	X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	Y = [i[1] for i in training_data]
	return X, Y