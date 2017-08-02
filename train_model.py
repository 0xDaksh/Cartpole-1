from model import create_model
from process_data import process_data
def create_and_train_model(training_data, outputnodes, learning_rate, dir):
	X, Y = process_data(training_data)
	net = create_model(len(X[0]), outputnodes, learning_rate, dir)
	net.fit({'input': X}, {'targets': Y}, n_epoch = 3, snapshot_step=500, 
														  show_metric=True, run_id='openai-f1')
	net.save('saves/model')
	return net