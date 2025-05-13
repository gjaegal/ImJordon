import numpy as np

def rollout_trajectory(env, policy, max_traj_length, render=False):

    # initialize env
    ob, _ = env.reset()

    obs, acs, rews, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render image of the simulated env
        if render:
            img = env.render()
            image_obs.append(img)

        # use the most recent ob to decide what to do
        obs.append(ob)
        if policy is None:
            # if no policy is provided, take a random action
            ac = env.action_space.sample()
        else:
            ac = policy.get_action(ob)
        acs.append(ac)
        # take that action and record results
        ob, rew, terminated, truncated, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rews.append(rew)

        # end the rollout if the rollout ended
        rollout_done = (terminated or truncated) or (steps >= max_traj_length)  
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Traj(obs, image_obs, acs, rews, next_obs, terminals)

def rollout_n_trajectories(env, policy, ntraj, max_traj_length, render=False):
    """
    Collect ntraj rollouts.
    """
    trajs = []
    for _ in range(ntraj):
        traj = rollout_trajectory(env, policy, max_traj_length, render)
        trajs.append(traj)

    return trajs

def Traj(obs, image_obs, acs, rewards, next_obs, terminals):
    """
    Take info (separate arrays) from a single rollout
    and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }