import gym
import random 
import numpy as np
import tflearn
from statistics import mean, median
from collections import Counter
from tqdm import tqdm
from createData import initial_population
from train_model import create_and_train_model 
from model import create_model
from process_data import process_data
# import this for testing gym
#from randomGamesFirst import random_games_first

# vars
LR = 1e-4
env = gym.make('CartPole-v0')
env.reset()
env = gym.wrappers.Monitor(env, '/tmp/cartpolev0-daksh-experiment-1', force=True)
goal_steps = 1000
score_requirement = 50
initial_games = int(1e+4)


# to get training data use initial_population()
#initial_population(env, goal_steps, score_requirement, initial_games)
training_data = np.load('training_data.npy')

# Create and train model
#model = create_and_train_model(training_data, 2, LR, 'log')
# I've trained the model and saved weights, hence I'm not training any more
X, Y = process_data(training_data)
model = create_model(len(X[0]), 2, LR, 'log')
model.load('saves/model')

# play game with model
scores = []
choices = []
for each_game in range(100):
	score = 0
	game_memory = []
	prev_obs = []
	env.reset()
	for _ in range(goal_steps):
		env.render()
		if len(prev_obs) == 0:
			action = random.randrange(0, 2)
		else:
			action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
		choices.append(action)
		new_observation, reward, done, info = env.step(action)
		prev_obs = new_observation
		game_memory.append([new_observation, action])
		score += reward
		if done:
			break
	scores.append(score)

print('Average Scores', sum(scores) / len(scores))
print('Choice 1: {}'.format(choices.count(1) / len(choices)))
print('Choice 2: {}'.format(choices.count(0) / len(choices)))