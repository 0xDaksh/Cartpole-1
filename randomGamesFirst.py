def random_games_first(env, goal_steps):
	for episode in range(5):
		env.reset()
		for t in range(goal_steps):
			env.render()
			action = env.action_space.sample()
			observations, reward, done, info = env.step(action)
			if done:
				break