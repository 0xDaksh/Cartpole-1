import tflearn

def create_model(input_size, output_nodes, learning_rate, dir):
	nn = tflearn.input_data(shape=[None, input_size, 1], name='input')

	nn = tflearn.fully_connected(nn, 128, activation='relu')
	nn = tflearn.dropout(nn, 0.8)

	nn = tflearn.fully_connected(nn, 256, activation='relu')
	nn = tflearn.dropout(nn, 0.8)

	nn = tflearn.fully_connected(nn, 512, activation='relu')
	nn = tflearn.dropout(nn, 0.8)

	nn = tflearn.fully_connected(nn, 256, activation='relu')
	nn = tflearn.dropout(nn, 0.8)

	nn = tflearn.fully_connected(nn, 128, activation='relu')
	nn = tflearn.dropout(nn, 0.8)

	nn = tflearn.fully_connected(nn, 256, activation='relu')
	nn = tflearn.dropout(nn, 0.8)
	if output_nodes > 1:
		activation = 'softmax'
	else:
		activation = 'sigmoid'
	nn = tflearn.fully_connected(nn, output_nodes, activation=activation)
	nn = tflearn.regression(nn, optimizer='adam', loss='categorical_crossentropy', 
								learning_rate=learning_rate, name="targets")
	return tflearn.DNN(nn, tensorboard_dir=dir)