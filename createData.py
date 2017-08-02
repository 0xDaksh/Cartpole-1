import numpy as np
from statistics import mean, median
from collections import Counter
import random
def initial_population(env, goal_steps, score_requirement, initial_games):
	training_data = []
	scores = []
	accepted_scores = []
	for _ in range(initial_games):
		score = 0
		game_memory = []
		prev_observation = []
		for _ in range(goal_steps):
			action = random.randrange(0, 2)
			observation, reward, done, info = env.step(action)
			if len(prev_observation) > 0:
				game_memory.append([prev_observation, action])
			prev_observation = observation
			score += reward
			if done:
				break
		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1] == 1:
					output = [0, 1]
				elif data[1] == 0:
					output = [1, 0]
				training_data.append([data[0], output])
		env.reset()
		scores.append(score)
	training_data_save = np.array(training_data)
	np.save('training_data.npy', training_data_save)
	print('Average accepted score: ', mean(accepted_scores))
	print('Median Accepted Score: ', median(accepted_scores))
	print(Counter(accepted_scores))
	return training_data