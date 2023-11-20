import gym

env = gym.make("SpaceInvaders-v4",render_mode = "human")

env.reset()

terminated = False


while not terminated:
    new_step = env.action_space.sample()
    new_state,reward,terminated, truncated,info = env.step(new_step)
    env.render()



 

    # env.step(env.action_space.sample())
    # env.render()